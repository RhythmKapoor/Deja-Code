# Déjà Code — Semantic Search for Competitive Programming

**Find problems by what they mean, not just by the words they contain.**

Déjà Code is a problem-discovery tool built around a simple idea: when practising competitive programming, finding a problem similar to one you have already solved can be just as useful as finding a completely new one.

Traditional keyword searches work well when the wording matches. But programming problems often describe the same underlying idea in different ways. A problem about counting factors, for example, might be related to another problem involving divisors even if their titles and statements use different words.

Déjà Code explores this problem using text processing, semantic embeddings, information retrieval and clustering. It provides an interactive interface where users can search a collection of competitive programming problems, compare retrieval methods and explore groups of related questions.

---

## Table of Contents

- [The Problem](#the-problem)
- [What Déjà Code Does](#what-déjà-code-does)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Search Methods](#search-methods)
- [Clustering and Visualisation](#clustering-and-visualisation)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Using the Application](#using-the-application)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Acknowledgements](#acknowledgements)

## The Problem

Competitive programming platforms contain problems that frequently overlap in their underlying concepts, even when the statements look quite different.

Consider a few example queries:

- "Count the number of divisors of an integer."
- "Find how many factors a number has."
- "Determine the number of positive integers that divide N."

A keyword-based search may focus on the exact words used in the query. A semantic search system attempts to capture the similarity in meaning between the query and the available problem statements.

This distinction matters when someone wants to:

- Find practice questions related to a problem they have already solved.
- Discover alternative problems involving a familiar algorithmic idea.
- Explore a topic without knowing the exact terminology used by the platform.
- Identify recurring patterns across a collection of programming problems.

The goal of Déjà Code is to make this kind of exploration easier by comparing different retrieval approaches and presenting their results through a simple web interface.

## What Déjà Code Does

The current implementation brings together four main capabilities:

**1. Search by keywords**

A TF-IDF-based search method retrieves problems using the importance of terms in the query and problem text.

**2. Search by meaning**

A sentence-transformer model converts text into numerical embeddings. These representations allow the system to compare the semantic similarity between a query and a problem.

**3. Hybrid retrieval**

A hybrid search mode combines lexical and semantic retrieval signals to produce a ranked list of candidate problems. This aims to balance exact terminology matches with similarity in meaning.

**4. Explore problem clusters**

The project groups problems into clusters using K-Means and provides a visual representation of those groups. This offers another way to investigate relationships within the dataset, beyond searching for a single query.

The application also displays useful problem information, including titles, problem identifiers, difficulty levels, tags, statement excerpts and links to the corresponding CodeChef problems when available.

## Dataset

The project uses a collection of **1,430 competitive programming problems**.

The dataset contains fields such as:

| Field | Purpose |
|---|---|
| `QCode` | Problem identifier |
| `Title` | Problem title |
| `level` | Difficulty information |
| `statement` | Original problem description |
| `Tags` | Available problem tags |
| `clean_statement` | Processed statement used in text analysis |

The dataset is cleaned and prepared before being used for retrieval and clustering. Text fields are particularly important because the quality of the search results depends on how well the problem descriptions represent the underlying tasks.

The project retains both problem-level information and processed text so that the search methods can work with textual representations while the interface can still display readable problem details.

*Dataset note: the repository contains the processed data files used by the application. The original raw dataset is not included in the current repository snapshot.*

## How It Works

The workflow can be understood as a sequence of data preparation, representation, retrieval and exploration steps.

### 1. Data inspection and preprocessing

The first step is to inspect the dataset, identify the relevant columns and prepare the text for analysis.

The preprocessing workflow creates a cleaned dataset containing a processed statement field. This provides a consistent text input for the subsequent retrieval and clustering stages.

The original title, tags, identifiers and other metadata are retained because they are useful for displaying and interpreting the results.

### 2. TF-IDF representation

TF-IDF stands for *Term Frequency–Inverse Document Frequency*.

It assigns numerical weights to terms based on their occurrence in a document and their relative rarity across the collection. Terms that help distinguish one document from another can therefore contribute more strongly to its representation.

The resulting vectors allow the system to compare a search query with problem descriptions using cosine similarity.

TF-IDF provides a useful lexical baseline: it helps establish how well a conventional text retrieval approach performs before semantic information is introduced.

### 3. Semantic embeddings

Déjà Code uses the `sentence-transformers/all-MiniLM-L6-v2` model to generate dense vector representations of text.

Unlike a basic bag-of-words representation, an embedding is intended to capture aspects of the meaning and context of a text passage.

In this project, the generated embeddings have **384 dimensions**. The resulting vectors can be compared to estimate how closely a query relates to a problem description.

The generated embeddings are cached locally in `data/semantic_embeddings.npy` to avoid repeating the embedding-generation step unnecessarily during subsequent runs.

The model is useful for discovering related descriptions that do not necessarily share all the same keywords. However, semantic similarity is not proof that two problems require the same algorithm or have identical difficulty.

### 4. Retrieval and ranking

When a user submits a query, the application uses the selected search method to retrieve and rank candidate problems.

The results are presented in a readable format so that users can inspect the problem title, difficulty, tags and statement excerpt rather than relying on a numerical score alone.

The three search modes serve different purposes:

- **TF-IDF:** prioritises lexical relationships between the query and the problem text.
- **Semantic:** uses embedding-based similarity to find problems related in meaning.
- **Hybrid:** combines lexical and semantic retrieval signals to produce a ranked list.

Comparing these methods makes it possible to observe where keyword matching is sufficient and where semantic retrieval may provide more useful results.

## Search Methods

### TF-IDF search

TF-IDF provides a transparent and computationally practical baseline. It can perform well when a query uses terminology that also appears in the relevant problem statement.

Its main weakness is that different wording can hide a meaningful relationship. It may also rank a problem highly because it shares prominent words with a query, even when the underlying programming task is not especially relevant.

### Semantic search

Semantic retrieval uses dense embeddings generated by the sentence-transformer model.

This allows the system to compare textual meaning beyond exact word overlap. For instance, a query about the factors of a number may retrieve a problem that describes counting its divisors.

Semantic search is not guaranteed to understand algorithmic intent perfectly. A problem can be linguistically similar to a query without requiring the same solution technique.

### Hybrid search

Hybrid retrieval brings the two approaches together. TF-IDF contributes lexical relevance, while semantic retrieval contributes meaning-based relevance.

The aim is to reduce dependence on either signal alone and provide a more useful overall ranking.

The hybrid mode is also useful for comparison: its results can be inspected alongside the other two methods to see whether combining retrieval signals improves the practical relevance of the retrieved problems.

## Clustering and Visualisation

Search is useful when a user already has a question in mind. Clustering provides a complementary way to explore the dataset without starting with a specific query.

The current clustering pipeline uses **TF-IDF features and K-Means with eight clusters**.

K-Means groups data points according to their positions in a numerical feature space. In this project, the TF-IDF representation provides the features used to group the problem descriptions.

To make the groups easier to inspect visually, Truncated Singular Value Decomposition (Truncated SVD) reduces the high-dimensional text representation to a lower-dimensional space for plotting.

The resulting visualisation provides an overview of how the problems are distributed across the clusters.

These clusters should be treated as exploratory groups rather than definitive algorithm categories. The method does not automatically guarantee that every problem in a cluster uses the same data structure, algorithm or programming technique.

## Evaluation

A retrieval system should not be judged only by whether its results look plausible. It should also be evaluated against examples where relevance has been assessed.

Déjà Code includes a small manually labelled evaluation set and a script that calculates two ranking metrics.

### Evaluation metrics

**Precision@5**

Precision@5 measures the proportion of the first five retrieved results that are labelled relevant.

\[
\text{Precision@5} =
\frac{\text{Relevant results in top 5}}{5}
\]

A higher value indicates that a greater proportion of the top-ranked results are relevant.

**nDCG@5**

Normalised Discounted Cumulative Gain at 5 evaluates the relevance of the top five results while giving greater importance to relevant results appearing earlier in the ranking.

It is useful because the order of search results matters: finding a relevant problem in the first position is generally more helpful than finding it at the bottom of the list.

### Preliminary results

The current manually labelled evaluation produced the following results:

| Retrieval method | Precision@5 | nDCG@5 |
|---|---:|---:|
| Semantic | 0.520 | 0.850 |
| TF-IDF | 0.400 | 0.762 |

On this evaluation sample, semantic retrieval performed better than the TF-IDF baseline on both metrics.

The Precision@5 result for semantic search indicates that an average of 2.6 out of the top five results were labelled relevant across the evaluated queries. For TF-IDF, the corresponding average was 2 out of five.

These are preliminary findings, not proof that semantic search will outperform TF-IDF for every query or on every dataset. The manual evaluation contains only five unique queries, so a larger and more diverse test set is needed to draw stronger conclusions.

The evaluation scripts and labelled examples are included in the repository so that the results can be inspected and the evaluation can be extended.

## Project Structure

```text
Deja-Code/
│
├── app.py
├── clean_data.py
├── clustering.py
├── compare_search.py
├── evaluate_manual.py
├── evaluate_search.py
├── inspect_data.py
├── prepare_evaluation.py
├── real_search.py
├── search_engine.py
├── semantic_search.py
│
├── data/
│   ├── cleaned_questions.csv
│   ├── clustered_questions.csv
│   ├── clusters.png
│   └── manual_evaluation.csv
│
├── requirements.txt
└── README.md
```

The exact files present may change as the project evolves. The repository also excludes the local Python virtual environment and generated embedding cache.

### Main files

- `app.py` — Streamlit interface for searching problems and exploring clusters.
- `clean_data.py` — prepares and cleans the dataset.
- `search_engine.py` — contains the core search-engine functionality.
- `semantic_search.py` — handles semantic embedding generation and related search functionality.
- `clustering.py` — performs K-Means clustering and generates the cluster visualisation.
- `compare_search.py` — supports comparison of retrieval approaches.
- `prepare_evaluation.py` — prepares evaluation examples.
- `evaluate_manual.py` — evaluates manually labelled search results.
- `evaluate_search.py` — supports search evaluation.
- `real_search.py` — provides additional search functionality and experimentation.
- `inspect_data.py` — helps inspect the dataset and its fields.

## Getting Started

### Requirements

- Python 3.11 is recommended for the environment used to develop this project.
- pip
- Git, if cloning the repository
- An internet connection on the first run may be needed to download the sentence-transformer model.

### 1. Clone the repository

```bash
git clone https://github.com/RhythmKapoor/Deja-Code.git
cd Deja-Code
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the Command Prompt activation script instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The application uses Python libraries for data processing, machine learning, semantic embeddings and the web interface.

### 4. Launch the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL in the terminal. Open that URL in your browser to use the application.

The first launch may take longer because the sentence-transformer model may need to be downloaded and loaded. Keep the required processed data files in the `data/` directory.

If you need to regenerate embeddings, run the relevant embedding-generation script:

```bash
python semantic_search.py
```

The original raw dataset is not included in the repository, so reproducing the entire preprocessing pipeline from scratch requires obtaining that dataset separately.

## Using the Application

### Search Problems

1. Enter a natural-language query describing the problem or concept you want to explore.
2. Select TF-IDF, Semantic or Hybrid search.
3. Inspect the ranked results.
4. Compare titles, problem identifiers, difficulty, tags and statement excerpts.
5. Open the linked CodeChef problem, when a link is available, to examine the complete statement.

Try queries that describe the same concept in different words to compare the behaviour of the retrieval methods.

### Explore Clusters

Open the cluster exploration section to inspect the available problem groups and their visual representation.

Use the grouping as a starting point for exploration rather than assuming that every problem in a cluster has an identical solution approach.

### About

The About section provides context about the project, its purpose and the methods used to build the application.

## Limitations and Future Improvements

The current version is a working prototype, and there are several areas where it can be improved.

**Larger evaluation set.** The manually labelled sample is small. More queries and independent relevance judgements would make the comparison between retrieval methods more reliable.

**Better hybrid ranking.** Different combinations of lexical and semantic scores or ranks could be tested systematically. Their performance should be compared on the same evaluation set rather than assuming that a hybrid method is automatically better.

**Algorithm-aware representations.** The current semantic model represents natural-language text; it is not specifically trained to recognise algorithmic equivalence. Incorporating reliable problem tags, editorial information, solution approaches or specialised embeddings could improve this aspect.

**Improved clustering.** The eight-cluster configuration is an initial choice. Alternative cluster counts, representations and clustering algorithms could be compared, with both quantitative measures and manual inspection.

**More extensive error analysis.** Relevant and irrelevant results should be inspected to identify recurring false positives, missed matches and differences between retrieval methods.

**Larger datasets and efficiency.** The application could be extended to more problems while improving indexing, caching and retrieval speed.

**More complete testing and deployment.** Automated tests, clearer data-version tracking and repeatable deployment checks would make the project easier to maintain.

These improvements would help establish not only whether the system can retrieve related problems, but also which kinds of relationships it handles well and where it still struggles.

## Acknowledgements

Déjà Code was developed as a learning project focused on information retrieval, natural-language processing and practical machine learning.

The implementation uses open-source Python libraries, including pandas, scikit-learn, sentence-transformers and Streamlit.

The semantic representation model used in the project is `sentence-transformers/all-MiniLM-L6-v2`.

---

**Déjà Code — making it easier to discover the next problem worth solving.**
