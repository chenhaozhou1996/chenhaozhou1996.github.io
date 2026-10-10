# Homework 2: Reading and Drawing a Simple Regression

Demand Planning & Fulfillment  
Posted October 10, 2026 · **Due Saturday, October 17, 2026**

In this homework, use one X variable to describe or predict one Y variable. Read a regression table, draw a best-fit straight line, and run a small analysis in Google Colab. No confidence intervals, hypothesis tests, or multiple regression are required.

**Estimated time: 90–120 minutes.** Question 1: about 20 minutes; Question 2: about 20 minutes; Question 3, including Colab and writing: about 50–70 minutes. Analyze only one dataset for Question 3.

## Question 1: Read the regression table

A store wants to understand the relationship between its daily customer count (X) and units sold (Y). The following results were calculated from the six-day **simulated teaching dataset** in `store_practice.csv`.

| Term | Coefficient |
| --- | ---: |
| Intercept | 19.600 |
| Customers (X) | 3.011 |

| Model information | Value |
| --- | ---: |
| Observations | 6 |
| R-squared | 0.999 |

1. Identify X and Y and give their units.
2. Write the fitted equation in the form: predicted Y = intercept + slope × X.
3. Explain the slope in your own words. What does one additional customer mean for predicted units sold?
4. Explain what the intercept represents. Is a prediction at zero customers necessarily useful when the observed data begin at ten customers?
5. Predict units sold on a day with 35 customers. Use the coefficients shown above.
6. Explain R-squared in plain language. Does a high R-squared mean that every prediction is exact?

## Question 2: Draw the best-fit line

Use the same store data:

| Customers (X) | Units sold (Y) |
| ---: | ---: |
| 10 | 52 |
| 20 | 78 |
| 30 | 110 |
| 40 | 139 |
| 50 | 168 |
| 60 | 203 |

1. Draw a scatterplot with clearly labeled axes and units. You may draw it by hand or use Colab.
2. Use your equation from Question 1 to calculate two points on the fitted line, at X = 10 and X = 60. Draw the straight line through those two fitted points on your scatterplot. Do not connect the observed dots to one another.
3. Explain why some observed points lie above the line and others below it. What does the line help the store understand?

## Question 3: Choose data and explain your own regression

Find a CSV dataset that interests you, or choose **one** of the original CSV files provided with this homework. You do not need to analyze every file. Use two numerical columns: one X and one Y. If you find your own data, include its source link and briefly describe where it came from. Choose data with at least 20 usable rows and more than one distinct X value.

| Downloadable original CSV | What a row represents | Suggested X | Suggested Y | Units |
| --- | --- | --- | --- | --- |
| `tips.csv` | A restaurant bill | `total_bill` | `tip` | Both in dollars |
| `mpg.csv` | A car model | `weight` | `mpg` | Pounds; miles per gallon |
| `penguins.csv` | A penguin | `flipper_length_mm` | `body_mass_g` | Millimeters; grams |

These three files are unchanged copies from the [seaborn example-data repository](https://github.com/mwaskom/seaborn-data). They are historical practice data. Some cells are missing; the notebook removes rows only when your selected X or Y is missing or nonnumeric. The store file is simulated teaching data.

### Start from the original files

1. Download `HW2_regression_starter.ipynb` and your chosen CSV to your computer. If you download the ZIP, unzip it first.
2. Open [Google Colab](https://colab.research.google.com/) and sign in with your Google account. The free standard runtime is sufficient.
3. Choose **File → Upload notebook**, then select `HW2_regression_starter.ipynb`. If Colab shows an opening dialog, select its Upload tab.
4. Click **Connect** if needed. Each code cell has a triangular Run button on its left. Run the cells in order from top to bottom.
5. Run the CSV upload cell. Click **Choose Files** and select your CSV. This uploads the data separately from the notebook.
6. Run the preview cell. Read the column names and first rows. In the next cell, change `x_column` and `y_column` to match the two columns you chose, keeping the quotation marks. For `mpg.csv`, use `weight` and `mpg`; for `penguins.csv`, use `flipper_length_mm` and `body_mass_g`.
7. Run the remaining cells to obtain the coefficient table, model-information table, and scatterplot with a fitted straight line. The notebook also creates `regression_coefficients.csv`, `regression_model_info.csv`, and `regression_plot.png` for download.
8. Use **File → Download → Download .ipynb** to save your completed notebook with outputs. Upload the CSV again if you start a new Colab session.

### Your explanation

Include your data source and chosen variables, the regression tables, the fitted equation, and your scatterplot with the fitted line. Then explain, in your own words:

- What question are you asking, and what do X and Y measure?
- What does the slope mean, using the correct units? What does the intercept mean in this setting?
- What does R-squared tell you about the fit?
- Pick an X value inside your observed range. Calculate the predicted Y and explain what it means.
- Describe one practical use of this line and one reason predictions may be imperfect. Explain why an association by itself does not show that changing X causes Y to change.

Do not simply paste the program output as your explanation. Your words should refer to your particular data and results.

## What to submit

Submit either a **Word document (.doc or .docx)** or **handwritten answers** (clear photos or a scan). Answer Questions 1–3 in your own words. For Question 3, run Colab yourself and include **screenshots of both regression tables and the scatterplot with the fitted line**. Insert screenshots into your Word document, or attach them alongside your handwritten pages. Label each screenshot and refer to it in your explanation. Include source links for independently found data. You do not need to submit the notebook or CSV. Clear interpretation matters more than programming experience.

## Help with common problems

- **Column-name error:** copy the column name exactly from the preview, including capitalization and underscores.
- **No upload button yet:** wait for the upload cell to finish starting, then select your CSV.
- **Missing or nonnumeric values:** the notebook reports how many rows were removed for your chosen columns. Use the remaining sample count in your explanation.
- **Disconnected session:** reconnect and run the cells from the beginning, uploading the CSV again.

Colab upload and notebook-download instructions follow the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
