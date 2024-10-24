import os
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import common
from custom_logger import CustomLogger
from logmod import logs

logs(show_level='info', show_color=True)
logger = CustomLogger(__name__)  # use custom logger

# set template for plotly output
template = common.get_configs('plotly_template')
response = common.get_configs("response")
directory_path = common.get_configs("output")


def age_distribution(response, output_folder):
    # Check if df is a string (file path), and read it as a DataFrame if necessary
    if isinstance(response, str):
        df = pd.read_csv(response)

    # Count the occurrences of each age
    age_counts = df.groupby('What is your age?').size().reset_index()
    age_counts.columns = ['What is your age?', 'count']

    # Convert the 'What is your age (in years)?' column to numeric (ignoring errors for non-numeric values)
    age_counts['What is your age?'] = pd.to_numeric(age_counts['What is your age?'], errors='coerce')

    # Drop any NaN values that may arise from invalid age entries
    age_counts = age_counts.dropna(subset=['What is your age?'])

    # Sort the DataFrame by age in ascending order
    age_counts = age_counts.sort_values(by='What is your age?')

    # Extract data for plotting
    age = age_counts['What is your age?'].tolist()
    counts = age_counts['count'].tolist()

    # Add ' years' to each age label
    age_labels = [f"{int(a)} years" for a in age]  # Convert age values back to integers

    # Create the pie chart
    fig = go.Figure(data=[
        go.Pie(labels=age_labels, values=counts, hole=0.0, showlegend=True, sort=False)
    ])

    # Update layout
    fig.update_layout(
        legend_title_text="Age"
    )

    # Save the figure in different formats
    base_filename = "age"
    fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
    fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
    fig.write_image(os.path.join(output_folder, base_filename + ".svg"), width=1600, height=900, scale=3, format="svg")
    pio.write_html(fig, file=os.path.join(output_folder, base_filename + ".html"), auto_open=True)


def gender_distribution(df, output_folder):
    # Check if df is a string (file path), and read it as a DataFrame if necessary
    if isinstance(df, str):
        df = pd.read_csv(df)
    # Count the occurrences of each gender
    gender_counts = df.groupby('What is your gender?').size().reset_index()
    gender_counts.columns = ['What is your gender?', 'count']

    # Drop any NaN values that may arise from invalid gender entries
    gender_counts = gender_counts.dropna(subset=['What is your gender?'])

    # Extract data for plotting
    genders = gender_counts['What is your gender?'].tolist()
    counts = gender_counts['count'].tolist()

    # Create the pie chart
    fig = go.Figure(data=[
        go.Pie(labels=genders, values=counts, hole=0.0, marker=dict(colors=['red', 'blue', 'green']),
               showlegend=True)
    ])

    # Update layout
    fig.update_layout(
        legend_title_text="Gender"
    )

    # Save the figure in different formats
    base_filename = "gender"
    fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
    fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
    pio.write_html(fig, file=os.path.join(output_folder, base_filename + ".html"), auto_open=True)


def scene_columns(df, output_folder):
    # Check if df is a string (file path), and read it as a DataFrame if necessary
    if isinstance(df, str):
        df = pd.read_csv(df)

    # Find all columns that match the base name "Which image looks more like a real picture of a driving scene?"
    driving_scene_columns = [col for col in df.columns
                             if "Which image looks more like a real picture of a driving scene?" in col]

    # Loop through all occurrences of the column and generate charts
    for idx, col_name in enumerate(driving_scene_columns):
        # Access the specific occurrence of the column
        driving_scene_column = df[col_name]

        # Count the occurrences for the current column
        driving_scene_counts = driving_scene_column.value_counts().reset_index(name='count')
        driving_scene_counts.columns = ['Which image looks more like a real picture of a driving scene?', 'count']

        # Drop any NaN values that may arise
        driving_scene_counts = driving_scene_counts.dropna()

        # Extract data for plotting
        responses = driving_scene_counts['Which image looks more like a real picture of a driving scene?'].tolist()
        counts = driving_scene_counts['count'].tolist()

        # Create the pie chart
        fig = go.Figure(data=[
            go.Pie(labels=responses, values=counts, hole=0.0, marker=dict(colors=['red', 'blue', 'green']),
                   showlegend=True)
        ])

        # Update layout
        fig.update_layout(
            legend_title_text="Response"
        )

        # Save the figure in different formats (using a unique filename for each occurrence)
        base_filename = f"driving_scene_column_{idx + 1}"
        fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
        fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
        pio.write_html(fig, file=os.path.join(output_folder, base_filename + ".html"), auto_open=True)


def calculate_mean_std(df):
    # Check if df is a string (file path), and read it as a DataFrame if necessary
    if isinstance(df, str):
        df = pd.read_csv(df)

    # Find all columns that match the base name "On a scale of 1 to 10,
    # how would you rate the realism of the image you chose above?"
    rating_columns = [col for col in df.columns
                      if "On a scale of 1 to 10, how would you rate the realism of the image you chose above?" in col]

    # Loop through all occurrences of the column and calculate the mean and standard deviation
    for idx, col_name in enumerate(rating_columns):
        # Calculate the mean and standard deviation for the specific column
        mean_value = df[col_name].mean()
        std_value = df[col_name].std()

        # Print the result with the index
        print(f"Stats for column {idx + 1}: '{col_name}'")
        print(f"  Mean: {mean_value}")
        print(f"  Standard Deviation: {std_value}")
        print("-" * 40)


try:
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
except Exception as e:
    print(f"Error occurred while creating directory: {e}")

age_distribution(response, directory_path)
gender_distribution(response, directory_path)
scene_columns(response, directory_path)
calculate_mean_std(response)
