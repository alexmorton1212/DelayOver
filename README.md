# Flight Delay Analyzer

A Streamlit web application to explore and analyze flight delays using historical airline performance data from the U.S. Department of Transportation.  

The app allows users to visualize patterns in flight delays over the years and gain insights into airline performance.

## 🚀 Live Demo

Check out the app live: [DelayOver Streamlit App](https://delayover.streamlit.app)

## 📊 Data Source

This project uses the **On-Time Performance Data** provided by the Bureau of Transportation Statistics (BTS):

[On-Time Reporting: Carrier On-Time Performance (1987 - Present)](https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present)

## 🛠 Technology Stack

- **Python 3.12**  
- **Streamlit** for interactive web interface  
- **Docker & Docker Compose** for containerized deployment  
- **Pandas / NumPy / Plotly / Matplotlib** (optional, depending on your visualizations)  

## 💻 Installation & Usage

You can run the app locally either with Docker or by running Streamlit directly.

### Option 1: Run with Docker

1. Build the Docker image:

```bash
docker build -t delayover_app .
docker-compose up --build
```