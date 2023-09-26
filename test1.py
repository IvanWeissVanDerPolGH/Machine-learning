from functions import *

# Generate scattered points and create DataFrames
num_points = 500
group_names_color = [{"name": 'Group_1', "color": 'yellow'}, {"name": 'Group_2', "color": 'Blue'}]

points_data_group1 = generate_scattered_points(num_points, center_x=10, center_y=0, min_dist=0, max_dist=10)
df_group1 = create_dataframe(points_data_group1, group_names_color[0])

points_data_group2 = generate_scattered_points(num_points, center_x=0, center_y=20, min_dist=0, max_dist=10)
df_group2 = create_dataframe(points_data_group2, group_names_color[1])

# Concatenate the DataFrames for both groups
df_points = pd.concat([df_group1, df_group2], ignore_index=True)

# Train SVM models and plot decision boundaries
trained_models = train_svm_model_for_groups(df_points)
# plot_groups_with_decision_boundary(df_points, trained_models)

while True:
    
    classify_new_point(df_points,trained_models,trained_models['Group_1']['model'], group_names_color)

