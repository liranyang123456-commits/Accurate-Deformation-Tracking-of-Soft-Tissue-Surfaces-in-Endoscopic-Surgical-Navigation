@echo off
setlocal
cd /d "%~dp0"

python candidate_exporter.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python run_nested_oof.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python evaluate_oof.py
if errorlevel 1 exit /b %errorlevel%

python make_figures.py
if errorlevel 1 exit /b %errorlevel%

python visual_feature_exporter.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python run_visual_oof.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python evaluate_visual_oof.py
if errorlevel 1 exit /b %errorlevel%

python train_final_visual.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python visual_feature_exporter_dynamic.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python run_visual_oof.py --config config/default.yaml --dynamic
if errorlevel 1 exit /b %errorlevel%

python evaluate_visual_oof.py --visual artifacts/visual_oof_predictions_dynamic.csv --manifest artifacts/visual_run_manifest_dynamic.json --responses artifacts/visual_responses_dynamic.npz --out-dir artifacts/visual_dynamic_evaluation --model-name visual_dynamic --hybrid-name hybrid_dynamic_router
if errorlevel 1 exit /b %errorlevel%

python train_final_visual.py --config config/default.yaml --dynamic
if errorlevel 1 exit /b %errorlevel%

python run_visual_oof.py --config config/default.yaml --dynamic --experiment-tag size --size-weight 0.15
if errorlevel 1 exit /b %errorlevel%

python temporal_safety_oof.py
if errorlevel 1 exit /b %errorlevel%

python mask_targets.py --config config/default.yaml
if errorlevel 1 exit /b %errorlevel%

python run_mask_oof.py
if errorlevel 1 exit /b %errorlevel%

python train_final_mask.py
if errorlevel 1 exit /b %errorlevel%

python expanded_candidates.py
if errorlevel 1 exit /b %errorlevel%

python export_trackves_dense_nano.py --config config/default.yaml --sequences EV1,EV2,EV3,IV1,IV2,IV3,IV4,IV5,IV6
if errorlevel 1 exit /b %errorlevel%

python dense_consistency_oof.py
if errorlevel 1 exit /b %errorlevel%

python external_sequence_adapter.py config/external_sequence_validation_test.json --known EV1,EV2,EV3,IV1,IV2,IV3,IV4,IV5,IV6,capture_photos11,captured_photos1 --out artifacts/external_sequence_validation.json
if errorlevel 1 exit /b %errorlevel%

python evaluate_advanced.py
if errorlevel 1 exit /b %errorlevel%

echo Learned-fusion OOF experiment complete.
