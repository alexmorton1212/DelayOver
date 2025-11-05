
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors

# --------------------------------------------------------------------------------------------------------
# DIRECTORIES
# --------------------------------------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data', 'processed')
PARQUET_FILE = os.path.join(PROCESSED_DATA_DIR, 'summary_dataset.parquet')

df = pd.read_parquet(PARQUET_FILE)

print(df.head())

# --------------------------------------------------------------------------------------------------------
# TITLE PAGE
# --------------------------------------------------------------------------------------------------------

def create_title_page():

    fig, ax = plt.subplots(figsize=(8.5, 11))  # US Letter size
    ax.axis("off")  # Hide axes

    # Big centered title
    fig.text(0.5, 0.7, "My App Update for October 2025",
             ha='center', va='center', fontsize=24, weight='bold')

    # Subtitle or description
    fig.text(0.5, 0.6, "Summary of metrics, improvements, and key charts",
             ha='center', va='center', fontsize=14)

    # Add a footer or small text
    fig.text(0.5, 0.1, "Generated automatically by Python",
             ha='center', fontsize=10, color='gray')

    return fig


# --------------------------------------------------------------------------------------------------------
# OVERVIEW PAGE
# --------------------------------------------------------------------------------------------------------

### TBD

# --------------------------------------------------------------------------------------------------------
# ORIGIN PAGE
# --------------------------------------------------------------------------------------------------------

### TOP 40 ON-TIME PERCENTAGE (SHOW 75TH, 90, 95TH, 99TH PERCENTILE)

def create_flight_summary_figure(df, title="Flight Delay Summary"):

    df['origin_ui'] = df['origin_ui'].str[:3]

    # Aggregate
    summary = df.groupby('origin_ui').agg(
        total_flights=('if_delay', 'count'),
        delayed_flights=('if_delay', 'sum'),
        p75=('arrdelayminutes', lambda x: np.percentile(x, 75)),
        p90=('arrdelayminutes', lambda x: np.percentile(x, 90)),
        p95=('arrdelayminutes', lambda x: np.percentile(x, 95)),
        p99=('arrdelayminutes', lambda x: np.percentile(x, 99))
    )

    summary['on_time_pct'] = 1 - summary['delayed_flights'] / summary['total_flights']
    summary = summary[['on_time_pct', 'p75', 'p90', 'p95', 'p99']]
    summary = summary.sort_values('on_time_pct', ascending=False)

    # Move origin_ui into first column and rename to 'Airport'
    summary_reset = summary.reset_index()
    summary_reset.rename(columns={'origin_ui': 'Airport'}, inplace=True)

    # Color scales for numeric columns only
    cmap_on_time = plt.cm.RdYlGn
    cmap_percentiles = plt.cm.RdYlGn_r
    numeric_cols = ['on_time_pct', 'p75', 'p90', 'p95', 'p99']
    norms = {col: colors.Normalize(vmin=summary_reset[col].min(), vmax=summary_reset[col].max()) 
             for col in numeric_cols}

    # Prepare cell colors
    cell_colors = []
    for _, row in summary_reset.iterrows():
        row_colors = [
            'white',  # Airport column stays white
            cmap_on_time(norms['on_time_pct'](row['on_time_pct'])),
            cmap_percentiles(norms['p75'](row['p75'])),
            cmap_percentiles(norms['p90'](row['p90'])),
            cmap_percentiles(norms['p95'](row['p95'])),
            cmap_percentiles(norms['p99'](row['p99']))
        ]
        cell_colors.append(row_colors)

    # Column headers
    columns = ['Airport'] + numeric_cols

    # Create figure
    fig_width = 8
    fig_height = len(summary_reset)*0.25 + 1.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    cell_text = summary_reset.copy()
    cell_text[numeric_cols] = cell_text[numeric_cols].round(2)  # round only numeric columns

    table = ax.table(
        cellText=cell_text.values,
        colLabels=columns,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center'
    )

    # Adjust font and scaling
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(0.9, 1.2)

    plt.title(title, fontsize=14, pad=12)
    plt.tight_layout()

    return fig


# --------------------------------------------------------------------------------------------------------
# AIRLINE PAGE
# --------------------------------------------------------------------------------------------------------


# Suppose 'fig' is the figure returned by your create_flight_summary_figure function
origin_fig = create_flight_summary_figure(df)
title_fig = create_title_page()

# Export to PDF
pdf_path = "flight_summary.pdf"

with PdfPages(pdf_path) as pdf:

    pdf.savefig(title_fig)
    plt.close(title_fig)

    pdf.savefig(origin_fig, bbox_inches='tight', pad_inches=0.5)  # keeps figure centered
    plt.close(origin_fig)

# plt.title("Flight Delay Summary by Origin Airport", fontsize=14, pad=12)
# plt.tight_layout()
# plt.show()



# --------------------------------------------------------------------------------------------------------
# ORIGIN & DESTINATION CONNECTIONS PAGE
# --------------------------------------------------------------------------------------------------------

### TOP 10 CONNECTIONS BASED ON ON-TIME PERCENTAGE (SHOW 75TH, 90, 95TH, 99TH PERCENTILE)
### BOTTOM 10 CONNECTIONS BASED ON ON-TIME PERCENTAGE (SHOW 75TH, 90, 95TH, 99TH PERCENTILE)



# with PdfPages("app_update_report.pdf") as pdf:

#     # --- PAGE 1: TITLE PAGE ---
#     fig, ax = plt.subplots(figsize=(8.5, 11))  # US Letter size
#     ax.axis("off")  # Hide axes

#     # Big centered title
#     fig.text(0.5, 0.7, "My App Update for October 2025",
#              ha='center', va='center', fontsize=24, weight='bold')

#     # Subtitle or description
#     fig.text(0.5, 0.6, "Summary of metrics, improvements, and key charts",
#              ha='center', va='center', fontsize=14)

#     # Add a footer or small text
#     fig.text(0.5, 0.1, "Generated automatically by Python",
#              ha='center', fontsize=10, color='gray')

#     pdf.savefig(fig)
#     plt.close(fig)

#         # --- PAGE 2: EXAMPLE PLOT ---
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot([1, 2, 3, 4], [10, 12, 8, 15], marker='o')
#     ax.set_title("Daily Active Users", fontsize=16)
#     ax.set_xlabel("Day")
#     ax.set_ylabel("Users")
#     pdf.savefig(fig)
#     plt.close(fig)

#     # --- PAGE 3: TEXT PAGE ---
#     fig, ax = plt.subplots(figsize=(8.5, 11))
#     ax.axis("off")

#     report_text = (
#         "Highlights:\n\n"
#         "- User engagement increased by 12%.\n"
#         "- Crash rate decreased by 30%.\n"
#         "- New feature X had positive feedback.\n\n"
#         "Next Steps:\n"
#         "- Continue monitoring feature adoption.\n"
#         "- Prepare A/B test for November release."
#     )
#     fig.text(0.1, 0.8, report_text, va='top', fontsize=12)
#     pdf.savefig(fig)
#     plt.close(fig)