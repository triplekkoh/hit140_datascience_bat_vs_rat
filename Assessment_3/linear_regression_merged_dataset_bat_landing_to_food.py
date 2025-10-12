# run_linear_with_selector.py
from model_select import pick_for_script, print_ranking_table
from linear_workflow import run_linear_workflow

# Choose dataset + candidates
dataset_path = "merged_dataset_with_rat_eating.csv"
response_var = "bat_landing_to_food"
candidate_predictors = ['seconds_after_rat_arrival', 'risk', 'reward', 
                        'rat_minutes',
                        'food_availability', 'rat_eating', 
                        'rat_arrival_number', 
                        'rat_presence_duration_sec']

# Pick best subset by test R²
response_var, predictor_vars, dataset_path, ranking = pick_for_script(
    dataset_path,
    response_var,
    candidate_predictors,
    test_size=0.40,
    random_state=42,
)

print_ranking_table(ranking, top_n=5)

# Run the full workflow (plots included)
results = run_linear_workflow(
    data=dataset_path,
    response_var=response_var,
    predictor_vars=predictor_vars,
    test_size=0.40,
    random_state=42,
    clip_negative_preds=True,
    show_plots=True,
)
