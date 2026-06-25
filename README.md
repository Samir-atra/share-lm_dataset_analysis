# ShareLM Dataset Analysis

This repository contains a collection of Python scripts for analyzing, studying, and optimizing the [ShareLM dataset](https://huggingface.co/datasets/shachardon/ShareLM), which is a collection of human-model chat conversations. The analysis focuses on understanding the distribution of models, languages, user contributions, and conversation lengths, and enriches the dataset with topic classifications using a Gemma model.

## Project Structure

The project is organized into the following modules within the `src` directory:

-   `src/adding_topic`: Contains scripts for adding topic classifications to the dataset using a Gemma model.
-   `src/utils`: Provides utility functions for data handling, API quota management, and file operations.
-   `src/visualizations`: Includes scripts to generate various plots for visualizing the dataset analysis.

## Key Features

*   **Dataset Loading:** Loads the ShareLM dataset from Hugging Face.
*   **Data Analysis:** Performs analysis on various aspects of the dataset, including:
    *   **Model Usage:** Counts and visualizes the frequency of different models used in the conversations.
    *   **Language Distribution:** Analyzes the distribution of languages in the dataset.
    *   **User Contributions:** Identifies and visualizes the top contributors to the dataset.
    *   **Conversation Length:** Analyzes the distribution of conversation lengths.
*   **Topic Modeling:** Uses a Gemma model to classify conversations into predefined topics.
*   **Data Visualization:** Generates various plots to visualize the analysis results, including bar charts, histograms, and scatter plots.
*   **Data Export:** Saves the processed dataset with topic classifications to a CSV file.

## Visualizations

The scripts in `src/visualizations` generate the following plots:

*   A horizontal bar chart showing the top 20 most frequent models, with a subplot of a scatter plot showing individual model counts.
*   A horizontal bar chart showing the frequency of models with names (excluding the most used model).
*   A horizontal bar chart showing the frequency of languages.
*   A horizontal bar chart showing the top users by contribution count.
*   A horizontal histogram showing the distribution of conversation lengths.
*   A more detailed horizontal histogram showing the distribution of conversation lengths between 0 and 1000.

## Setup and Usage

### Environment Setup

This project is set up to run in a development container. The `.devcontainer/devcontainer.json` file specifies the required Docker image and dependencies.

The following Python dependencies are required:
* `huggingface_hub`
* `datasets`
* `pandas`
* `google-generativeai`
* `google-colab`
* `transformers`
* `torch`
* `tensorflow`
* `seaborn`
* `matplotlib`

These dependencies are automatically installed when the dev container is created.

### Running the Scripts

1.  Open the project in a dev container-compatible editor (e.g., VS Code with the Dev Containers extension).
2.  **API Keys:** The scripts require API keys for Hugging Face and Google AI. You will need to set these up as environment variables:
    - `HF_TOKEN`: Your Hugging Face API token.
    - `GOOGLE_API_KEY`: Your Google AI API key.
3.  **Running the Analysis:** The analysis can be performed by running the scripts in the `src` directory. The main script for processing the dataset is `src/adding_topic/add_topic.py`. The visualization scripts can be run to generate the plots.

## Dataset

The analysis is performed on the **ShareLM dataset**. You can find more information about the dataset on its [Hugging Face page](https://huggingface.co/datasets/shachardon/ShareLM).

## Results

The analysis reveals several key findings:

*   The most used model in the dataset is "N/A", which indicates that the conversation was collected from another dataset and not using the ShareLM plugin.
*   The top 20 most frequent models include "N/A" and several named models, with counts decreasing sharply after the top few. The most used models are GPT, with a preference for the latest versions.
*   The dataset contains conversations in multiple languages, with English being the dominant language.
*   User contributions are highly skewed, with a few users contributing a large number of conversations.
*   Conversation lengths vary widely, with a large number of short conversations (0-1000 turns) and a long tail of much longer conversations.
*   Approximately 10,000 conversations in the dataset were collected using the plugin, while the remaining ~300,000 are from other datasets.

## References

1.  Don-Yehiya S, Choshen L, Abend O. The ShareLM collection and plugin: contributing human-model chats for the benefit of the community. arXiv preprint arXiv:2408.08291. 2024 Aug 15.
2.  Meyer S, Elsweiler D. " You tell me": a dataset of GPT-4-based behaviour change support conversations. InProceedings of the 2024 Conference on Human Information Interaction and Retrieval 2024 Mar 10 (pp. 411-416).
3.  Zhao W, Ren X, Hessel J, Cardie C, Choi Y, Deng Y. Wildchat: 1m chatgpt interaction logs in the wild. arXiv preprint arXiv:2405.01470. 2024 May 2.
4.  Hsu E, Yam HM, Bouissou I, John AM, Thota R, Koe J, Putta VS, Dharesan GK, Spangher A, Murty S, Huang T. WebDS: An End-to-End Benchmark for Web-based Data Science. arXiv preprint arXiv:2508.01222. 2025 Aug 2.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite it as follows:

Attrah, Samer,
Refining share lm: plugin expansion , synthetic annotation , and dataset analysis
(September 21, 2025). Available at SSRN: https://ssrn.com/abstract=5525145 or http://dx.doi.org/10.2139/ssrn.5525145 
