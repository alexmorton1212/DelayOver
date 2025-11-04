# DelayOver: Flight Delay Analyzer

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-%E2%89%A52.12-orange)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/docker-available-lightblue)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A Streamlit web application to explore and analyze flight delays using historical airline performance data from the U.S. Department of Transportation. Currently using August 2024 - July 2025 data for flights between the Top 40 U.S. Airports.  


## Live Demo

Check out the app live: 

- [DelayOver Streamlit App](https://delayover.streamlit.app)


## Data Source

This project uses the **On-Time Performance Data** provided by the Bureau of Transportation Statistics (BTS):

- [On-Time Reporting: Carrier On-Time Performance (1987 - Present)](https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present)

## Technology Stack

- **Python 3.12**  
- **Streamlit | HTML | CSS** for interactive web interface  
- **Docker | Docker Compose** for containerized deployment  
- **Pandas | NumPy | Plotly | Matplotlib** for data processing and visualizations
- **Scikit-Learn | XGBoost** for ML modeling

## Installation & Usage

You can run the app locally either with Docker or by running Streamlit directly.

### Option 1: Run with Docker

1. Build the Docker image:

```bash
docker build -t delayover_app .
docker-compose up --build
```

2. Open your browser at http://localhost:8501 to view the app.


### Option 2: Run with Python & Streamlit

1. Clone the repository:

```bash
git clone https://github.com/alexmorton1212/DelayOver.git
cd flight-delay-analyzer
```

2. Install dependencies (virtual environment recommended):

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open your browser at http://localhost:8501.