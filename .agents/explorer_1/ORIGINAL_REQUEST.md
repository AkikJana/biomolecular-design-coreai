## 2026-06-20T15:43:39Z
You are teamwork_preview_explorer. Your working directory is /Users/akikjana/Documents/BiomolecularDesign/.agents/explorer_1/. Your parent conversation ID is 3b22170c-2360-4307-8490-eadba5d7ed35.
Your mission is to investigate the Boltz structure prediction codebase and how it is run. Specifically:
1) Check the directory structure, identify CLI entrypoints, and look at src/predict_structure.py, boltz/src/boltz/model/models/boltz2.py, etc.
2) Figure out how to invoke predictions, what validation targets exist (e.g. insulin, hemoglobin, TNF-alpha), and where validation data is located.
3) Identify how the 4 features are implemented or integrated: MPS execution, Low-rank pair updates, CFG distillation, Neural refinement.
4) Write an exploration report to `/Users/akikjana/Documents/BiomolecularDesign/.agents/sub_orch_e2e_testing/exploration_report.md` detailing all of the above.
Also, initialize your own BRIEFING.md and progress.md in your working directory. Use send_message to notify the parent (me) when done.
