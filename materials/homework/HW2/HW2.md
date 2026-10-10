# Homework 2: Learn, Calculate, Draw, and Explain

Demand Planning & Fulfillment  
**Due Saturday, October 17, 2026 · Total: 100 points · Estimated time: 90–120 minutes**

There are three questions. Question 1 asks you to learn through your own conversation with AI, then calculate and draw a fitted line by hand. Question 2 asks you to upload data and obtain a fitted line in Google Colab. Question 3 asks you to interpret a provided graph. No confidence intervals, hypothesis tests, or multiple regression are required.

## Question 1: How can I calculate a best-fit line myself? — 50 points

**Open learning question:** If I want a formula for the best-fit straight line, rather than asking a computer to draw it for me, how can I calculate that line? How does the formula work, and why does it make sense?

### Learn through your own prompts

Use an AI assistant of your choice to learn the formula and method for a simple least-squares best-fit line with an intercept. Start with your own question and continue the conversation until you understand how to use the formula. Ask follow-up questions about anything you do not understand. There is no prescribed wording or required number of prompts.

Show the conversation that explains how you learned: include screenshots or a transcript of your actual prompts and the relevant AI replies. Explain **why you asked those questions**, what you learned from the replies, and how a follow-up helped resolve confusion or check your understanding. A pasted conversation by itself is not an answer. Write the formula you learned and explain its symbols and steps in your own words. You do not need to derive it using calculus.

### Apply your understanding by hand

Use these **five simulated teaching observations**. X is the number of workers on a shift; Y is the number of orders processed per hour.

| Observation | X: workers | Y: orders per hour |
| --- | ---: | ---: |
| A | 1 | 4 |
| B | 2 | 4 |
| C | 3 | 7 |
| D | 4 | 8 |
| E | 5 | 12 |

**Handwrite your calculation of the slope, intercept, and fitted equation.** Show your intermediate arithmetic so someone can follow how you used the formula. Explain how your calculation corresponds to the method you learned in the AI conversation.

On paper, draw and label the axes, plot all five observations, and **use a ruler to draw your calculated fitted straight line**. Show how you calculated two points on the line to position it. Do not connect the five observed dots to one another. A basic calculator may be used for arithmetic, but do not use AI or a regression program to calculate these five observations or produce this graph for you.

### Explain and evaluate the method

In your own words, explain what you think “best fit” means and what the formula is trying to achieve. Drawing on the class discussion, why is this a relatively reasonable way to choose a line? What problem does it solve that a less careful way of choosing or judging a line might create? Do you think this method is always reasonable? Explain at least one limitation or situation where you would question it, and give your reason. Also explain what your hand-drawn line says about these workers and orders.

Your response is open-ended: we are assessing the learning conversation, your own understanding, the handwritten application, and your reasoning about the method.

## Question 2: Upload, prompt, and fit a line in Colab — 30 points

Find a CSV dataset that interests you, or download **one** of the original files below. Choose one numerical X and one numerical Y. Analyze only one dataset for this question.

| Original CSV | A row represents | Suggested X | Suggested Y |
| --- | --- | --- | --- |
| `tips.csv` | A restaurant bill | `total_bill` (dollars) | `tip` (dollars) |
| `mpg.csv` | A car model | `weight` (pounds) | `mpg` (miles per gallon) |
| `penguins.csv` | A penguin | `flipper_length_mm` (mm) | `body_mass_g` (grams) |

The provided CSVs are unchanged historical practice files from the [seaborn example-data repository](https://github.com/mwaskom/seaborn-data). If you find your own data, include its source link.

Create a **blank Google Colab notebook** and upload the raw CSV yourself. Write your own prompts to Colab's AI assistant to read the file, fit a simple linear regression with an intercept, and display a regression table and a scatterplot with the fitted straight line. Request the intercept, slope, observation count, and R-squared. Ask for clearly labeled axes with units and no confidence band.

Include screenshots of the uploaded filename, your actual prompts, regression table, and graph. Explain in your own words:

- What question are you asking, what does one row mean, and why did you choose X and Y?
- Why did you write your prompts that way? What did the AI do to turn your uploaded data into the table and line? Explain any follow-up you needed.
- How do the slope, intercept, and line relate to the formula and method you learned in Question 1? Explain what the slope and R-squared mean for your data.
- Choose one specific statement in the AI's explanation. Do you agree, partly agree, or disagree? Refer to your table or graph to explain why. A correct statement is acceptable if you explain why it is correct.

### Colab starting steps

1. Download and, if necessary, unzip your CSV. Open [Google Colab](https://colab.research.google.com/), sign in, and create a new notebook.
2. Open the Gemini / AI chat using its spark icon. Attach the CSV through the chat's upload control and check the filename. Alternatively, upload through the Files panel and tell the AI the filename and runtime path shown there.
3. Write your own prompts. First check that the AI can read the correct file and show its columns; then ask for the analysis you need.
4. Review and run its proposed analysis or generated code cells to obtain actual output. Check the selected variables and labels, and follow up if something is wrong. Take screenshots of the results.

No ready-made regression notebook is supplied. If AI chat is unavailable, contact the instructor for help. No paid upgrade is required.

## Question 3: Explain the provided graph — 20 points

The graph below shows **simulated teaching data** for delivery distance and delivery time. The dots are eight observed deliveries; the straight line is their least-squares fitted line. Use the graph and small regression table to explain the relationship in your own words. You do not need Colab or a new analysis for this question.

![Delivery distance and time: observed dots and fitted line](HW2_reading_plot.png)

| Term | Coefficient |
| --- | ---: |
| Intercept | 9.857 |
| Distance (km) | 2.143 |

- What does the graph suggest about distance and delivery time? Explain the slope using its units.
- What is the difference between a dot and the line? Choose a dot above or below the line and explain what that position means.
- What could a delivery planner use this line for, and what should they avoid concluding from it? Explain one limitation visible in the graph or implied by these data.

## What to submit

Submit a **Word document (.doc or .docx)** or **handwritten answers** as clear photos or a scan, with the relevant screenshots inserted or attached. **Question 1's calculation and ruler-drawn graph must be handwritten**, even if the rest is typed.

Include the AI learning conversation and your explanation of it, the handwritten work, the Colab screenshots and your actual prompts, and your answers to the graph question. Your explanations of what you learned and how you understand the results must be your own. Screenshots or copied AI explanations alone do not complete the homework. You do not need to submit code or a notebook.

Colab upload and AI-chat guidance follows the [official Colab FAQ](https://research.google.com/colaboratory/faq.html).
