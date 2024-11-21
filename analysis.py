import os
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
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
answers = common.get_configs("answer")


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

    # Calculate mean and standard deviation of age
    mean_age = df['What is your age?'].mean()
    std_age = df['What is your age?'].std()

    logger.info(f"Mean Age: {mean_age}")
    logger.info(f"Standard Deviation of Age: {std_age}")

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


def demographics(df, output_folder):
    # Check if df is a string (file path), and read it as a DataFrame if necessary
    if isinstance(response, str):
        df = pd.read_csv(response)

    # Count the occurrences of each country
    country_counts = df['Which country are you currently in?'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']

    # Determine the number of unique countries
    num_countries = country_counts['Country'].nunique()
    print(f"Number of unique countries: {num_countries}")

    # Create the bar chart with different colors for each bar
    fig = px.bar(
        country_counts,
        x='Country',
        y='Count',
        title="Country Distribution",
        labels={'Country': 'Country', 'Count': 'Number of Responses'},
        color='Country',  # Different color for each country
        color_discrete_sequence=px.colors.qualitative.Safe  # Use a color palette
    )
    # Hide the legend
    fig.update_layout(showlegend=False)

    # Save the figure in different formats
    base_filename = "country_distribution"
    fig.write_image(os.path.join(output_folder, base_filename + ".png"), width=1600, height=900, scale=3)
    fig.write_image(os.path.join(output_folder, base_filename + ".eps"), width=1600, height=900, scale=3)
    fig.write_image(os.path.join(output_folder, base_filename + ".svg"), width=1600, height=900, scale=3, format="svg")
    fig.write_html(os.path.join(output_folder, base_filename + ".html"), auto_open=True)


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


def scene_columns_stacked_bar(df, output_folder, answers):
    # Check if df and answers are file paths, and read them if necessary
    if isinstance(df, str):
        df = pd.read_csv(df)
    if isinstance(answers, str):
        answers = pd.read_csv(answers)

    # Reset indexes for alignment
    answers = answers.reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Find all columns that match the repeated driving scene question pattern
    driving_scene_columns = [col for col in df.columns if
                             "Which image looks more like a real picture of a driving scene?" in col]

    # Ensure the number of question instances matches the answer rows
    if len(driving_scene_columns) != len(answers):
        raise ValueError("Mismatch between the number of question columns in `df` and rows in `answers`.")

    # Adjust answers in df based on comparison with answers.csv
    for i, col in enumerate(driving_scene_columns):
        df[col] = df.apply(
            lambda row: (
                "Ours" if row[col] == answers['Answers'].iloc[i] else
                ("Both images look alike" if row[col] == "Both images look alike" else "CycleGAN-turbo")
            ),
            axis=1
        )

    # Initialize a DataFrame to store counts of each answer type for each question column
    answer_counts = pd.DataFrame(columns=['Ours', 'CycleGAN-turbo', 'Both images look alike'],
                                 index=driving_scene_columns)
    for col in driving_scene_columns:
        counts = df[col].value_counts()
        answer_counts.loc[col, 'Ours'] = counts.get('Ours', 0)
        answer_counts.loc[col, 'CycleGAN-turbo'] = counts.get('CycleGAN-turbo', 0)
        answer_counts.loc[col, 'Both images look alike'] = counts.get('Both images look alike', 0)

    # Convert counts to percentages
    answer_counts = answer_counts.div(answer_counts.sum(axis=1), axis=0) * 100

    # Convert to long format for Plotly
    answer_counts_long = answer_counts.reset_index().melt(id_vars='index', var_name='Answer', value_name='Percentage')
    answer_counts_long['index'] = answer_counts_long['index'].apply(
        lambda x: f"Scene {driving_scene_columns.index(x) + 1}")
    answer_counts_long = answer_counts_long.rename(columns={'index': 'Question'})

    # Calculate mean and standard deviation of ratings
    stats = calculate_mean_std(df)

    # Create the stacked bar plot
    fig = px.bar(
        answer_counts_long,
        x='Question',
        y='Percentage',
        color='Answer',
        labels={'Question': 'Scenes', 'Percentage': 'Percentage (%)'},
    )
    fig.update_layout(barmode='stack', xaxis={'categoryorder': 'category ascending'})

    # Add mean and standard deviation annotations for each scene
    for scene, stat in stats.items():
        fig.add_annotation(
            x=scene,
            y=101,  # Positioning above the 100% bar
            text=f"M: {stat['mean']:.2f}, SD: {stat['std']:.2f}",
            showarrow=False,
            font=dict(size=16)
        )

    # Customize the layout
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Percentage of participants selecting each option for scene realism",
        yaxis_title_font=dict(size=18),
        xaxis_tickangle=0
    )

    # Set tick font size for x and y axes
    fig.update_xaxes(tickfont=dict(size=16))
    fig.update_yaxes(tickfont=dict(size=16))

    # Save the figure in different formats
    fig.write_image(os.path.join(output_folder, "stacked_bar_plot.png"), width=1600, height=900, scale=3)
    fig.write_image(os.path.join(output_folder, "stacked_bar_plot.eps"), width=1600, height=900, scale=3)
    fig.write_html(os.path.join(output_folder, "stacked_bar_plot.html"), auto_open=True)


def calculate_mean_std(df):
    # Check if df is a string (file path), and read it as a DataFrame if necessary
    if isinstance(df, str):
        df = pd.read_csv(df)

    # Find all columns that match the base name for the realism rating question
    rating_columns = [col for col in df.columns
                      if "On a scale of 1 to 10, how would you rate the realism of the image you chose above?" in col]

    # Calculate mean and std for each rating column
    stats = {}
    for idx, col_name in enumerate(rating_columns):
        mean_value = df[col_name].mean()
        std_value = df[col_name].std()
        stats[f"Scene {idx + 1}"] = {"mean": mean_value, "std": std_value}
    return stats


# age_distribution(response, directory_path)
# gender_distribution(response, directory_path)
# demographics(response, directory_path)
# scene_columns(response, directory_path)
scene_columns_stacked_bar(response, directory_path, answers)
# calculate_mean_std(response)
