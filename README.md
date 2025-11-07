# Online-Learning-Trends
Data cleaning and Power BI dashboard analyzing global online courses. Includes ETL in Python, interactive filters, KPIs, and insights on pricing, ratings, duration, and skills.


## Project Overview
This project analyzes the **Kaggle Online Courses dataset** to uncover insights into online learning trends and course performance.

The workflow includes **data cleaning (Python)**, **ETL preparation**, and an **interactive Power BI dashboard** for data visualization and insight generation.

---

## Dataset
- **Source:** [Kaggle — Online Courses Dataset](https://www.kaggle.com/datasets/khaledatef1/online-courses)  
- **Attributes Extracted:**
  - Category  
  - Rating  
  - Price  
  - Duration (hours)  
  - Skills  
  - Number of Viewers  
  - Level, Language, Site, and Program Type  

---

## Data Preparation
Data cleaning and preprocessing were handled using **Python (pandas, numpy, re, langdetect, deep_translator)**.  
Key steps:
- Removed duplicates and handled missing values  
- Standardized numeric and text fields  
- Converted prices and durations to numeric formats  
- Translated non-English text using Google Translate API  
- Extracted key attributes and validated field consistency  

The final cleaned dataset was exported for use in Power BI.

---

## Dashboard Features
Developed using **Power BI**, the dashboard explores multiple perspectives on online courses:

### **Main Components**
- **KPIs:**  
  - Total Courses  
  - Average Rating  
  - Median Price  
  - Average Duration (hours)  
  - % of Free Courses  

- **Visuals:**  
  - Top Categories (by course count & rating)  
  - Course Rating Distribution by Category  
  - Price vs Rating Scatter Plot  
  - Duration vs Rating Comparison  
  - Average Ratings   
  - Course Trends by Program Type  

- **Filters / Slicers:**  
  - Category  
  - Language  
  - Site  
  - Level  
  - Price Range  



