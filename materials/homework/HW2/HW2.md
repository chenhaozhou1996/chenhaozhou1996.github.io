# Homework 2: Upload, Prompt, Fit, and Explain

Demand Planning & Fulfillment  
**Due Saturday, October 17, 2026 · Estimated time: 90–120 minutes**

Practice taking a raw CSV file through a simple linear regression: upload data, choose X and Y, obtain a regression table and fitted straight line, check the output, and explain your decisions. You may use Colab's AI assistant or another method you choose in Colab. If you use AI, explain both your prompts and what the AI actually did. No prior coding experience is required.

**The task is to demonstrate your understanding of the process and results. Screenshots or an AI-written explanation alone do not complete the homework.** No confidence intervals, hypothesis tests, or multiple regression are required.

## Question 1: Read a regression table (about 15 minutes)

A store studies daily customer count (X, people per day) and units sold (Y, units per day). These results come from the six-day **simulated teaching dataset** in `store_practice.csv`.

| Term | Coefficient |
| --- | ---: |
| Intercept | 19.600 |
| Customers (X) | 3.011 |

| Model information | Value |
| --- | ---: |
| Observations | 6 |
| R-squared | 0.999 |

1. Write the fitted equation: predicted Y = intercept + slope × X.
2. Explain the slope and intercept in your own words, with units. Is zero customers inside the observed range of 10–60 customers? What does that mean for interpreting the intercept?
3. Use the displayed coefficients to calculate predicted units sold at 35 customers. Show your substitution and arithmetic.
4. Explain R-squared in plain language. Does this value mean every observed dot lies exactly on the line?

## Question 2: Complete the workflow yourself (about 30–40 minutes)

Download `store_practice.csv`. Create a **blank Google Colab notebook**, upload the CSV yourself, and obtain a regression table and a scatterplot with a fitted straight line. Use `customers` as X and `units_sold` as Y.

You may ask Colab's AI to generate and run the analysis, or use your own method. For the AI route, write your own prompts. No completed regression notebook is supplied.

Include screenshots showing your uploaded filename, regression table, and fitted-line plot. If you use AI, also show your actual prompts. Then answer:

1. **Explain your actions:** What did you upload? What does one row represent? Explain the steps you took from the file to the table and graph, and why each step was needed.
2. **Explain your instructions:** If you used AI, explain why you wrote your prompt that way. Identify two specific choices in your wording, such as naming X and Y, requesting a straight line, or asking for axis units. What did each choice help the AI do? Describe any follow-up and why you needed it. If you did not use AI, explain how your chosen method produced the table and fitted line.
3. **Check the result yourself:** Compare the fitted intercept and slope with Question 1. Use your calculation for 35 customers to check the fitted line at X = 35; state whether it agrees approximately. Explain why this is a check of the fitted line rather than an observed day's actual sales. Small differences from displayed rounding are acceptable.
4. **Interpret the line:** In your own words, what does it tell the store? Why do some dots lie above or below it?

### Getting started in Colab

1. Download the CSV to your computer. If using the ZIP, unzip it first.
2. Open [Google Colab](https://colab.research.google.com/), sign in, and create a **new notebook**.
3. Open the **Gemini / AI chat** panel using its spark icon. Attach your CSV using the chat's upload control and check the filename. Alternatively, upload through Colab's Files panel and tell the AI the filename and runtime path shown there.
4. Start by asking it to read your uploaded file and display the column names and first rows. Check that they match your data.
5. Write your own request for a simple linear regression with an intercept. Name the file, X, and Y. Request the intercept, slope, observation count, R-squared, fitted equation, and a scatterplot with the fitted straight line. Specify axis labels and units, and ask for no confidence band. Ask it to report any rows removed for missing X or Y.
6. Review and run the proposed analysis. If it generates code cells, run them to obtain actual output. Check the variables and labels; use a follow-up prompt to correct anything wrong.

A prompt is an instruction in everyday language. One possible first request is: **"Read the CSV I uploaded and show its column names and first five rows."** Write the analysis request yourself, using your selected variables. You may use the language you are comfortable with. Keep the prompts you actually used.

## Question 3: Choose data, explain your choices, and check the interpretation (about 45–60 minutes)

Find a CSV that interests you, or choose **one** of the provided original files below. Use one numerical X and one numerical Y, with at least 20 usable rows and varying X. Upload this file yourself and repeat the analysis in Colab. Analyze only one dataset for this question.

| Original CSV | A row represents | Suggested X | Suggested Y |
| --- | --- | --- | --- |
| `tips.csv` | A restaurant bill | `total_bill` (dollars) | `tip` (dollars) |
| `mpg.csv` | A car model | `weight` (pounds) | `mpg` (miles per gallon) |
| `penguins.csv` | A penguin | `flipper_length_mm` (mm) | `body_mass_g` (grams) |

These are unchanged historical practice files from the [seaborn example-data repository](https://github.com/mwaskom/seaborn-data). Some values are missing. If you find your own data, include the source link.

Include screenshots of your regression table and fitted-line plot. Answer the following **in your own words**, referring to your actual results:

1. **Your question and decisions:** What do you want to learn? What does one row represent? Why did you choose these variables, and why is one X and the other Y? Give their units.
2. **Your process and prompts:** Explain how you obtained the results. If you used AI, include the actual prompts, explain the purpose of your main instructions, and describe what the AI did with the data, including any removed rows. Explain a follow-up if you used one. If you used another method, explain your choices and steps instead.
3. **Your interpretation:** Write the fitted equation and explain the slope, intercept, and R-squared in the setting of your data. Describe one practical use of the line and one limitation. Do not simply copy the AI's explanation.
4. **Your numerical check:** Choose one actual row from the CSV. Show its X and observed Y. Use the fitted equation to calculate predicted Y, then calculate observed Y minus predicted Y. Check whether that dot should be above or below the line, and compare with the graph. Explain what this difference means. You may use a calculator.
5. **Your judgment of an explanation:** If you used AI, quote one specific sentence from its explanation of your results. Do you agree, partly agree, or disagree? Support your judgment with the data, table, or plot, and rewrite the sentence in your own words if needed. Check, for example, whether it used the right units, confused prediction with an actual value, or claimed causation from an association. You do not need to find an error if the sentence is correct; explain why it is correct. If you did not use AI, select one claim in your own interpretation and explain what evidence supports it and what it cannot establish.

## What to submit

Submit a **Word document (.doc or .docx)** or **handwritten answers** as clear photos or a scan. Include your answers to Questions 1–3, the screenshots, your actual prompts if you used AI, and your numerical checks. Insert screenshots into the document or attach them beside the handwritten pages; label them so your explanations clearly refer to them.

You do not need to submit code or a notebook. AI may help perform the analysis and offer an explanation for you to evaluate. **Your explanations of your decisions, checks, and understanding must be written by you.** A submission with only output screenshots or copied AI prose is incomplete.

If an error occurs, send it to the AI in a follow-up and check the new output. Re-upload your CSV if a restarted session loses it. If AI chat is unavailable, ask the instructor for help or use another Colab method you can explain. No paid upgrade is required.

Colab upload and AI-chat guidance follows the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
