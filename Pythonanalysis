# Import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import httpx
import time

# Constants for API
API_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = "API Token Enter"

# Load data from the specified file path
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

# Example data processing function
def process_data(data):
    # Add your processing steps here
    if data is not None:
        print("Processing data...")
        processed_data = data.copy()
        return processed_data
    else:
        print("No data to process.")
        return None

# Visualization (example placeholder)
def plot_data(data):
    if data is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data.iloc[:, 0])  # Example plotting the first column
        plt.title("Example Plot")
        plt.xlabel("Index")
        plt.ylabel("Values")
        plt.show()
    else:
        print("No data to plot.")

# Query the LLM using the AI Proxy with retry logic
def query_llm_with_httpx(prompt, model="gpt-4o-mini", max_retries=5, retry_delay=30):
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": prompt},
        ],
    }

    for attempt in range(max_retries):
        try:
            response = httpx.post(API_URL, headers=headers, json=payload, timeout=30.0)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
            else:
                print(f"HTTP error occurred: {e}")
                break
        except httpx.RequestError as e:
            print(f"Request error occurred: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
    return None

# Generate Markdown narrative summarizing analysis and insights
def generate_markdown_story(analysis, visualizations):
    if not analysis or not isinstance(visualizations, list) or len(visualizations) == 0:
        print("Error: Invalid analysis or visualizations provided.")
        return None

    summary = f"""
    Dataset Shape: {analysis['shape']}
    Data Types: {analysis['data_types']}
    Missing Values: {analysis['missing_values']}
    Summary Statistics:
    {pd.DataFrame(analysis['summary_statistics']).to_string()}

    Visualizations:
    - Missing Values Heatmap: {visualizations[0] if len(visualizations) > 0 else 'Not Available'}
    """
    if len(visualizations) > 1:
        summary += f"\n- Pairplot: {visualizations[1]}"
    if len(visualizations) > 2:
        summary += f"\n- Correlation Heatmap: {visualizations[2]}"

    prompt = f"""
    Analyze the following dataset summary and visualizations:
    {summary}

    Write a Markdown narrative that:
    1. Describes the dataset briefly.
    2. Explains the analyses performed.
    3. Highlights key insights.
    4. Suggests implications or next steps based on the findings.
    Include references to the visualizations in your narrative.
    """
    return query_llm_with_httpx(prompt)

# Generate Markdown story and save it to a file
def step_4(data, analysis, visualizations):
    if data is None:
        print("Error: No data provided for Markdown generation. Please ensure Step 1 has been executed.")
        return
    if not analysis or not visualizations:
        print("Error: No analysis or visualizations available. Please ensure Steps 2 and 3 have been executed.")
        return

    print("\n--- Generating Markdown Story ---")
    markdown_story = generate_markdown_story(analysis, visualizations)
    if not markdown_story:
        print("Error generating Markdown story using the LLM.")
        return

    output_markdown_file = "README.md"
    try:
        with open(output_markdown_file, "w") as f:
            f.write(markdown_story)
        print(f"Markdown story saved to {output_markdown_file}.")
    except Exception as e:
        print(f"Error saving Markdown story to file: {e}")

# Main function for execution
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]

    # Load and process data
    data = load_data(file_path)
    processed_data = process_data(data)

    # Example analysis and visualizations (placeholders, replace with real steps)
    analysis = {
        "shape": processed_data.shape if processed_data is not None else None,
        "data_types": processed_data.dtypes.to_dict() if processed_data is not None else None,
        "missing_values": processed_data.isnull().sum().to_dict() if processed_data is not None else None,
        "summary_statistics": processed_data.describe().to_dict() if processed_data is not None else None,
    }
    visualizations = ["missing_values_heatmap.png", "pairplot.png", "correlation_heatmap.png"]

    # Plot data
    plot_data(processed_data)

    # Step 4: Generate Markdown story
    step_4(data, analysis, visualizations)

if __name__ == "__main__":
    main()
