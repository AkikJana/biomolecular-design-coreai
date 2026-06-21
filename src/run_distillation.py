import os
import sys
import time
import gc
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Ensure src is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from cfg_distillation import (
    TeacherVectorField,
    CFGDistilledVectorField,
    SyntheticStructureDataset,
    train_teacher_model,
    train_distilled_model,
    sample_trajectory_euler,
    initialize_student_from_teacher
)

class PeakMemoryTracker:
    """
    Tracks peak and current memory usage across CUDA, MPS, or CPU platforms.
    Allows measuring incremental activation memory by subtracting baseline memory.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.peak_mem = 0.0
        
    def get_current_mem(self) -> float:
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated()
        elif self.device.type == "mps":
            try:
                return torch.mps.current_allocated_memory()
            except Exception:
                return 0.0
        else:
            # CPU RAM
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == 'darwin':
                # macOS returns maxrss in bytes
                return float(usage)
            else:
                # Linux returns maxrss in kilobytes
                return float(usage * 1024)
                
    def reset(self):
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.device.type == "mps":
            torch.mps.empty_cache()
        self.peak_mem = self.get_current_mem()
        
    def update(self):
        self.peak_mem = max(self.peak_mem, self.get_current_mem())
                
    def get_peak_mb(self) -> float:
        return self.peak_mem / (1024 * 1024)
        
    def get_current_mb(self) -> float:
        return self.get_current_mem() / (1024 * 1024)

def run_experiment():
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Device Setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # 2. Hyperparameters
    node_dim = 64
    seq_dim = 16
    time_dim = 32
    batch_size = 64
    num_residues = 32
    num_train_samples = 1200
    num_test_samples = 256
    
    import argparse
    parser = argparse.ArgumentParser(description="Run CFG Distillation")
    parser.add_argument("--epochs", type=int, default=2, help="Number of student distillation epochs")
    parser.add_argument("--teacher_epochs", type=int, default=2, help="Number of teacher pre-training epochs")
    args, unknown = parser.parse_known_args()
    
    teacher_epochs = args.teacher_epochs
    student_epochs = args.epochs
    learning_rate = 1e-3
    guidance_range = (0.0, 4.0)
    
    # Create output directory for figures/artifacts
    output_dir = "/Users/akikjana/.gemini/antigravity-cli/brain/33b9e418-4fdc-4adc-9140-d7f59f76f970"
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Create Datasets and DataLoaders
    print("Generating synthetic helix datasets...")
    train_dataset = SyntheticStructureDataset(num_samples=num_train_samples, num_residues=num_residues, seq_dim=seq_dim)
    test_dataset = SyntheticStructureDataset(num_samples=num_test_samples, num_residues=num_residues, seq_dim=seq_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=num_test_samples, shuffle=False)
    
    # Get a batch of test data for evaluation
    test_coords, test_seq = next(iter(test_loader))
    test_coords = test_coords.to(device)
    test_seq = test_seq.to(device)
    
    # 4. Initialize Models
    print("Initializing models...")
    teacher = TeacherVectorField(node_dim=node_dim, seq_dim=seq_dim, time_dim=time_dim).to(device)
    student = CFGDistilledVectorField(node_dim=node_dim, seq_dim=seq_dim, time_dim=time_dim).to(device)
    
    # 5. Pre-train Teacher Model on standard flow matching target (dx = x1 - x0)
    teacher_loss = train_teacher_model(
        teacher=teacher,
        dataloader=train_loader,
        epochs=teacher_epochs,
        lr=learning_rate,
        device=device
    )
    
    # 6. Initialize Student from Pre-trained Teacher
    print("Initializing student weights from teacher backbone...")
    initialize_student_from_teacher(student, teacher)
    
    # 7. Perform CFG Distillation Training
    student_loss = train_distilled_model(
        teacher=teacher,
        student=student,
        dataloader=train_loader,
        epochs=student_epochs,
        lr=learning_rate,
        device=device,
        guidance_range=guidance_range,
        use_user_cfg_formula=True
    )
    
    # 8. Warmup runs to compile kernels & stabilize memory cache
    print("\nWarming up models on device to ensure fair benchmarking...")
    teacher.eval()
    student.eval()
    
    # Generate same random start noise for warmup and evaluation
    torch.manual_seed(999)
    x0 = torch.randn_like(test_coords).to(device)
    
    # Warmup runs
    _ = sample_trajectory_euler(teacher, x0[:16], test_seq[:16], s_val=1.5, steps=10, device=device, is_student=False)
    _ = sample_trajectory_euler(student, x0[:16], test_seq[:16], s_val=1.5, steps=10, device=device, is_student=True)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    print("Warmup complete.")
    
    # 9. Benchmarking
    print("\nBenchmarking and verifying distillation fidelity...")
    guidance_scales = [0.0, 1.5, 3.0]
    num_euler_steps = 20
    
    results = {}
    mem_tracker = PeakMemoryTracker(device)
    
    for s_val in guidance_scales:
        print(f"\n--- Testing Guidance Scale s = {s_val} ---")
        
        # --- Teacher (CFG: 2 passes per step) ---
        gc.collect()
        mem_tracker.reset()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            
        base_mem_teacher = mem_tracker.get_current_mb()
        start_time = time.perf_counter()
        
        xt_teacher = x0.clone()
        dt = 1.0 / num_euler_steps
        
        for step in range(num_euler_steps):
            t_val = step / num_euler_steps
            t = torch.full((num_test_samples,), t_val, device=device)
            
            # Forward pass 1: Conditional
            cond_mask_1 = torch.ones(num_test_samples, device=device)
            v_cond = teacher(xt_teacher, t, test_seq, cond_mask=cond_mask_1)
            mem_tracker.update()
            
            # Forward pass 2: Unconditional
            cond_mask_0 = torch.zeros(num_test_samples, device=device)
            v_uncond = teacher(xt_teacher, t, test_seq, cond_mask=cond_mask_0)
            mem_tracker.update()
            
            # CFG combination
            vt = v_cond + s_val * (v_cond - v_uncond)
            xt_teacher = xt_teacher + dt * vt
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
            
        end_time = time.perf_counter()
        teacher_time = end_time - start_time
        teacher_peak = mem_tracker.get_peak_mb()
        teacher_act_mem = max(0.0, teacher_peak - base_mem_teacher)
        
        # --- Student (Distilled CFG: 1 pass per step) ---
        gc.collect()
        mem_tracker.reset()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            
        base_mem_student = mem_tracker.get_current_mb()
        start_time = time.perf_counter()
        
        xt_student = x0.clone()
        s_tensor = torch.full((num_test_samples,), s_val, device=device)
        
        for step in range(num_euler_steps):
            t_val = step / num_euler_steps
            t = torch.full((num_test_samples,), t_val, device=device)
            
            # Single forward pass
            vt = student(xt_student, t, test_seq, s_tensor)
            mem_tracker.update()
            
            xt_student = xt_student + dt * vt
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
            
        end_time = time.perf_counter()
        student_time = end_time - start_time
        student_peak = mem_tracker.get_peak_mb()
        student_act_mem = max(0.0, student_peak - base_mem_student)
        
        # Calculate alignment metrics
        mse = torch.mean((xt_teacher - xt_student) ** 2).item()
        mae = torch.mean(torch.abs(xt_teacher - xt_student)).item()
        
        # Calculate speedup and memory savings
        speedup = teacher_time / student_time
        
        results[s_val] = {
            "teacher_time": teacher_time,
            "student_time": student_time,
            "speedup": speedup,
            "teacher_mem": teacher_act_mem,
            "student_mem": student_act_mem,
            "alignment_mse": mse,
            "alignment_mae": mae
        }
        
        print(f"Teacher wall-time: {teacher_time * 1000:.2f} ms | Peak Activation VRAM: {teacher_act_mem:.4f} MB")
        print(f"Student wall-time: {student_time * 1000:.2f} ms | Peak Activation VRAM: {student_act_mem:.4f} MB")
        print(f"Speedup Factor  : {speedup:.2f}x (Student is {100.0 * (teacher_time - student_time) / teacher_time:.1f}% faster)")
        print(f"Trajectory MSE  : {mse:.6f} | MAE: {mae:.6f}")
        
    # 10. Plot Results and Save Figures
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss curves
    axs[0].plot(teacher_loss, label="Teacher pre-training loss", color="blue", alpha=0.7)
    axs[0].plot(student_loss, label="Student distillation loss", color="orange", alpha=0.9)
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Huber Loss")
    axs[0].set_title("Training Loss Convergence")
    axs[0].grid(True, linestyle="--", alpha=0.6)
    axs[0].legend()
    
    # Plot 2: Speedup vs scale bar chart
    scales_str = [f"s={s}" for s in guidance_scales]
    t_times = [results[s]["teacher_time"] * 1000 for s in guidance_scales]
    s_times = [results[s]["student_time"] * 1000 for s in guidance_scales]
    
    x = range(len(guidance_scales))
    width = 0.35
    axs[1].bar([i - width/2 for i in x], t_times, width, label="Teacher CFG (2 passes/step)", color="#1f77b4")
    axs[1].bar([i + width/2 for i in x], s_times, width, label="Student Distilled (1 pass/step)", color="#ff7f0e")
    axs[1].set_ylabel("Inference Time (ms)")
    axs[1].set_title("Inference Speed Comparison (20 Euler steps)")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(scales_str)
    axs[1].grid(True, linestyle="--", alpha=0.6)
    axs[1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cfg_distillation_comparison.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved comparison plot to: {plot_path}")
    
    # Save text results for report
    results_path = os.path.join(output_dir, "benchmark_results.txt")
    with open(results_path, "w") as f:
        f.write("=== CFG Distillation Benchmark Results ===\n")
        f.write(f"Device: {device}\n")
        f.write(f"Number of test samples: {num_test_samples}\n")
        f.write(f"Number of Euler steps: {num_euler_steps}\n\n")
        for s_val, metrics in results.items():
            f.write(f"Guidance Scale s = {s_val}:\n")
            f.write(f"  Teacher time: {metrics['teacher_time'] * 1000:.2f} ms\n")
            f.write(f"  Student time: {metrics['student_time'] * 1000:.2f} ms\n")
            f.write(f"  Speedup: {metrics['speedup']:.2f}x\n")
            f.write(f"  Teacher peak activation memory: {metrics['teacher_mem']:.4f} MB\n")
            f.write(f"  Student peak activation memory: {metrics['student_mem']:.4f} MB\n")
            f.write(f"  Fidelity MSE: {metrics['alignment_mse']:.8f}\n")
            f.write(f"  Fidelity MAE: {metrics['alignment_mae']:.8f}\n\n")
            
    print(f"Saved benchmark text file to: {results_path}")

if __name__ == "__main__":
    run_experiment()
