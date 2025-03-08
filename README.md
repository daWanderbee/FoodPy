
# FoodPy

🍽️ Food Waste Prediction System

The Food Waste Prediction System is a Django-based web application that predicts food wastage based on various factors such as the number of plates consumed, footfall, and order details. The system uses a trained machine learning model to help restaurants and cafeterias minimize food waste by making data-driven decisions.
## Features

- 🔍 Food Waste Prediction based on past consumption trends
- 📊 Machine Learning Model Integration for accurate forecasting
- 📂 MinMaxScaler Normalization to preprocess input data
- 🌐 Django-based Web Interface for user interaction
- 🎨 Bootstrap/Tailwind Styling for an intuitive UI


## Tech Stack

**Client**: HTML, CSS, JavaScript, Bootstrap/TailwindCSS

**Server**: Django, Python

**Machine Learning**: Scikit-learn, XGBoost, Pandas, NumPy




## Installation

Clone the Repository

```bash
git clone https://github.com/daWanderbee/FoodPy.git
cd foodPy
```
Create a Virtual Environment

```bash
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate  # On Windows
```

Install Dependencies

```bash
pip install -r requirements.txt

```
Start the Development Server

```bash
python manage.py runserver
```
## 📌 Usage

Enter Input Values such as orders, footfall, and plates consumed.
Click Predict to get the estimated food wastage.
View the result on the response page.

## Screenshots

![App Screenshot](https://res.cloudinary.com/wanderbee/image/upload/v1741457531/derybi1xssgf6vcrn50f.png)

![App Screenshot](https://res.cloudinary.com/wanderbee/image/upload/v1741457532/pkb0zwrtnv6bvlemef88.png)
