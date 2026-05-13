# INFO 442 — M1 Project Proposal

## Project Title

Cross-Platform Analysis of Consumer Discussions on Chinese EV Brands

---

## Team Members

| Name (English) | Name (Chinese) | Student ID | Role |
|---|---|---|---|
| Jianwen Han | 韩建汶 | 320230941911 | Team Lead & Exploratory Data Analysis |
| Chuhao Wang | 王楚皓 | 320230942401 | Data Acquisition & Data Preprocessing |
| Ke Meng | 孟可 | 320230942231 | Exploratory Data Analysis & NLP-based modelling |
| Zhirun Han | 韩知润 | 320230941921 | Visualization & Dashboard Deployment |

---

# 1. Domain and Motivation

Electric vehicles (EVs) have become one of the most competitive and discussed industries in China. Major EV brands such as Tesla, BYD, and Xiaomi Auto are constantly discussed across Chinese social media platforms. Different platforms attract different user communities and communication styles, which may lead to significant differences in public perception, consumer concerns, and emotional reactions toward EV brands.

This project aims to analyze and compare discussions about EV brands across three major Chinese social media platforms:

- Douyin(TikTok)
- Bilibili
- Xiaohongshu(Rednote)

The project focuses on understanding how different online communities discuss EV brands differently, including sentiment, consumer concerns, and platform-specific discussion patterns.

We think this problem has practical business value because EV companies increasingly rely on social media analytics to:

- understand customer concerns,
- monitor brand reputation,
- improve marketing strategies,
- identify emerging consumer trends.

In addition, this project follows a complete data science workflow including data acquisition, preprocessing, exploratory data analysis, NLP-based modelling, visualization, and dashboard deployment.

---

# 2. Dataset Description

## Data Sources

The dataset will be collected from publicly accessible content on:

- Douyin(TikTok) - short video platform
- Bilibil - long video platform
- Xiaohongshu(Rednote) - young lifestyle sharing platform

The selected EV brands are:

- Tesla
- BYD
- Xiaomi Auto

Team members(Chuhao wang and Jianwen Han) has already completed a preliminary feasibility validation using MediaCrawler and confirmed that data acquisition from the selected platforms is technically feasible within the project scope.

---

## Expected Dataset Size

We expect to collect:

| Platform | Estimated Comments / Posts |
|---|---|
| Douyin | 500–1000 |
| Bilibili | 800–1,200 |
| Xiaohongshu | 300–500 |

The final dataset is expected to contain more than 2000 text entries and multiple structured features.

---

## Expected Features

The dataset may include:

- platform
- brand
- comment text
- publish time
- number of likes
- number of replies
- anonymized demographic metadata (e.g., gender and ip region)
- keyword tags

We will store the collected data that has removed personal privacy information in our project repository.

---

## Access Method

Data will be collected using:

- MediaCrawler Project(https://github.com/NanmiCoder)
- Python-based web scraping tools
- Publicly accessible social media pages

The project will maintain reproducibility by documenting the acquisition process and preprocessing pipeline in Jupyter notebooks.

---

## Ethical Issue

We will only collect publicly available data. No private user information, account credentials, or personally identifiable information will be stored or analyzed.

Usernames and user IDs will be excluded from the final dataset whenever possible. The project focuses on aggregate discussion patterns rather than individual users.

We will strictly abide by the user service agreements and privacy policies of each platform, and will only use the data for learning purposes and never for any real commercial uses.

---

# 3. Scientific Question

The main scientific question of this project is:

> How do consumer discussions and concerns about major EV brands differ across Chinese social media platforms?

The project specifically investigates:

1. Whether sentiment toward Tesla, BYD, and Xiaomi differs across platforms.
2. What topics and concerns are most frequently discussed for each EV brand.
3. Whether different social media communities emphasize different aspects of EV products, such as:
   - price,
   - design,
   - battery,
   - smart driving,
   - reliability,
   - brand image.
4. How platform characteristics influence online discussion patterns and engagement.

We aims to produce measurable insights using:

- sentiment analysis,
- topic frequency analysis,
- NLP-based classification models,
- visualization and comparative analysis.

These findings may help companies better understand platform-specific consumer behavior and communication strategies.

---

# 4. Preliminary Hypothesis

We hypothesize that:

1. Different platforms will exhibit significantly different discussion styles and sentiment patterns.

2. Bilibili users may focus more on:
   - technology,
   - performance,
   - specifications,
   - smart driving features.

3. Douyin discussions may contain more:
   - emotional reactions,
   - entertainment-oriented content,
   - short-form engagement behavior.

4. Xiaohongshu users may focus more on:
   - lifestyle,
   - design,
   - consumption experience,
   - personal recommendations.

5. Xiaomi Auto is expected to generate stronger emotional polarization due to its recent market popularity and online attention.

We also expect that platform differences may influence not only sentiment polarity but also the types of consumer concerns expressed online.

---

# 4. Preliminary Hypothesis

We hypothesize that consumer discussions about EV brands will differ significantly across Chinese social media platforms due to differences in platform culture, user demographics, and communication styles.

Specifically, we expect to find that:

1. Different platforms will exhibit distinct sentiment patterns and discussion focuses toward EV brands.

2. Bilibili users may focus more on:
   - technology,
   - vehicle performance,
   - specifications,
   - smart driving features,
   
because Bilibili has a relatively younger and technology-oriented user community, where users tend to engage in longer and more detailed discussions.

3. Douyin discussions may contain more:
   - emotional reactions,
   - entertainment-oriented content,
   - short-form engagement behavior,
   
because Douyin emphasizes short video content and fast user interaction, which may encourage more emotional and trend-driven discussions.

4. Xiaohongshu users may focus more on:
   - lifestyle,
   - vehicle design,
   - consumption experience,
   - personal recommendations,
   
because Xiaohongshu is strongly associated with lifestyle sharing, consumer decisions, and personal experience-based content.

5. We also expect different EV brands to generate different types of public reactions. In particular, Xiaomi Auto may show stronger emotional polarization due to its recent market popularity, strong online presence, and highly active public discussions.

Overall, we hypothesize that platform characteristics will influence not only sentiment polarity, but also the types of consumer concerns and discussion behaviors expressed online.

---

# 5. Team Roles

| Team Member | Responsibilities |
|---|---|
| Jianwen Han | Team leadership, project coordination, milestone management, GitHub repository management, exploratory data analysis (EDA), and final project integration |
| Chuhao Wang | Data acquisition, crawler setup, dataset collection, raw data management, and data preprocessing |
| Ke Meng | Exploratory data analysis (EDA), text cleaning, NLP-based modelling, sentiment analysis, and model evaluation |
| Zhirun Han | Data visualization, dashboard development, Streamlit deployment, and presentation material preparation |

All team members will collaboratively contribute to:
- report writing,
- presentation preparation,
- result interpretation,
- project discussion,
- and final project refinement.

---

# 6. Preliminary Technical Plan

Our initial plan is to use the following technologies:

- Python
- Pandas
- Scikit-learn
- Jieba/pkuseg
- Matplotlib / Seaborn
- Streamlit
- PyTorch
- XGBoost

The potential models we have chosen include(not yet verified):

- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machine (SVM)
- XGBoost
- TextCNN
- BiLSTM

Additionally, we may explore lightweight transformer-based or LLM-assisted methods (e.g., KIMI's latest transformer architecture) for semantic analysis and topic summarization.

---

# 7. Expected Deliverables

The project plans to deliver:

- A cleaned and reproducible dataset(main content is the commentary text)
- Jupyter notebooks documenting preprocessing and analysis
- 8 visualizations for EDA
- At least 2 NLP models with comparative evaluation
- A dashboard or demo
- A final report and presentation

---
