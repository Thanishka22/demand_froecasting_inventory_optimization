[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/swathi221103/demand_forecasting_inventory_optimization)


🧩 The Problem I Was Solving
Every retail business faces the same balancing act every single day — how much stock should I hold?
Order too much and you're bleeding money on warehouse space, spoilage, and tied-up capital. Order too little and you're turning away customers, losing revenue, and damaging brand trust through stockouts. Most small to mid-size retailers handle this with gut instinct or simple Excel spreadsheets — which means they're almost always wrong in one direction or the other.
I wanted to build something that approached this problem the way a data-driven operations team at a large retailer would: by forecasting demand at the individual store-product level, quantifying uncertainty in those forecasts, and converting that uncertainty into concrete, defensible inventory decisions.
The specific questions I wanted to answer were:

How much of each product will each store sell on any given day?
How confident are we in that forecast?
Given that uncertainty, how much safety stock should we hold?
At what inventory level should we trigger a new order?
How do those decisions change if we adjust our service level target or supplier lead time?


🏗️ How the Model Works
The Data
I simulated two years of daily retail sales data (Jan 2023 – Dec 2024) across 6 stores and 40 products — 175,200 store-product-day records in total. The simulation was designed to reflect real retail complexity:

Price elasticity — each product has a unique sensitivity to price changes, so demand responds realistically when prices move
Promotional effects — random promo days with discounts between 5–35%, plus a merchandising lift on top of the price effect
Seasonality — weekly patterns (weekends spike for beverages, snacks, frozen goods) and monthly sine-wave seasonality
Holiday events — Thanksgiving window (+15%), Holiday season (+20%), July 4th week (+10%), February events (+6%)
Supply disruptions — random days where replenishment orders arrive short, causing real stockouts
Inventory simulation — a simple Monday replenishment policy was run, so the recorded sales reflect actual availability, not just demand

This means the dataset has all the messiness of real retail data: stockouts that suppress observed sales below true demand, irregular promo cadences, and correlated seasonality across products.
Feature Engineering
Rather than building a separate model for each store-product pair (which would be 240 models and wouldn't generalize), I built a single global model trained on all store-product combinations simultaneously. The model learns shared patterns while using store and product identity as features.
Feature CategoryFeaturesLag featuresTrue demand 1 day ago, 7 days ago, 14 days agoRolling statistics7-day rolling mean demandTime featuresDay of week, month of yearPrice & promotionCurrent price, promo flag, discount percentageCategorical encodingsStore ID, product ID, product category, holiday event
The Model
Algorithm: XGBoost Regressor
Target: True daily demand (units) — not observed sales, to avoid stockout bias
Train period: Jan 2023 – Jun 2024 (18 months)
Test period: Jul 2024 – Dec 2024 (6 months holdout)
Architecture: 300 trees, max depth 6, learning rate 0.05, 80% subsampling
Training data:  129,600 rows × 61 features
Test data:       44,160 rows × 61 features
Global residual std: 4.14 units
Uncertainty Quantification
Instead of just producing point forecasts, I computed 95% prediction intervals using the standard deviation of training residuals — giving a range of plausible demand values for each day, not just a single number. This is what makes the inventory policy defensible rather than arbitrary.
Inventory Policy
With forecasts and uncertainty estimates in hand, I applied the standard statistical inventory formula:
Safety Stock  = Z × σ_demand × √(lead_time_days)
Reorder Point = μ_demand × lead_time_days + Safety Stock
Where:

Z = service level factor (1.28 for 90%, 1.65 for 95%, 1.88 for 97%, 2.33 for 99%)
σ_demand = forecast uncertainty (standard deviation of predictions)
μ_demand = average daily forecast
lead_time_days = days between placing and receiving an order

This is computed dynamically in the dashboard — change any parameter and every SKU's policy recalculates instantly.

😤 Challenges I Faced
Getting the deployment to work on Hugging Face Spaces was the most frustrating part of this project. The app works perfectly locally, but HF Spaces uses Docker under the hood and has very specific requirements that aren't obvious unless you've done it before:

HF Spaces requires port 7860 specifically — Streamlit defaults to 8501 and the app will silently fail without this change
The README.md must begin with a YAML metadata block on the very first line — even a single blank line before the --- causes a config error with no helpful error message
File paths inside Docker containers need to be resolved relative to the script using os.path.abspath(__file__), not assumed to be the current working directory

Feature leakage was another subtle issue. Early versions of the lag features were computed incorrectly across the full dataset before the train/test split, which caused unrealistically good test performance. I fixed this by computing lags strictly within each store-product group and being careful about the ordering of operations.
The stockout bias problem — observed sales are not the same as demand when items are out of stock. A product might have had demand of 50 units but only sold 30 because inventory ran out. Using sales as the target would teach the model to predict artificially low demand. I solved this by simulating and storing true demand separately from actual sales, and always training on true demand.

📊 Results & Insights
On the model:

The global XGBoost model achieves a residual standard deviation of 4.14 units across all store-product combinations — meaning most daily forecasts are within ~4 units of true demand
Promotional features and lag features were the strongest predictors of demand
Weekend uplift was strongest for Beverages, Snacks, and Frozen categories — consistent with consumer shopping behavior
Holiday windows showed clear and predictable demand spikes that the model learned to anticipate from the calendar features alone

On the inventory policy:

At a 95% service level with a 7-day lead time, the average SKU requires a safety stock of approximately 11 units
Moving from 95% to 99% service level increases safety stock by roughly 41% — a significant holding cost tradeoff for marginal service improvement
Shorter lead times have a larger impact on safety stock than higher service levels for most SKUs — meaning improving supplier reliability is often more cost-effective than simply holding more stock


💡 What I Learned
Technically:

How to build and scale a global forecasting model across many entities — a pattern used by companies like Amazon and Walmart at scale
The difference between point forecasting and probabilistic forecasting, and why uncertainty quantification is essential for operational decisions
How inventory theory connects to ML outputs — bridging the gap between a model's predictions and a business action
How to containerize and deploy a data science app with Docker, and the specific quirks of deploying to Hugging Face Spaces

About the problem domain:

Demand forecasting is only half the problem — translating forecasts into decisions is where the real business value lives
Service level is a business choice, not a technical one — the right answer depends on the cost of holding stock versus the cost of losing a sale, which varies by product and company
Real retail data is far messier than any simulation can capture — stockouts, returns, pricing errors, and data pipeline failures are all part of the challenge in production systems


🔮 Future Plans

 Model evaluation metrics — add RMSE, MAE, and MAPE on the test set to the dashboard so model accuracy is visible
 Baseline comparison — benchmark XGBoost against a naive 7-day moving average to show the model's added value
 Multi-step forecasting — extend from 1-day ahead to 7, 14, and 30-day forecast horizons
 SHAP explainability — show which features are driving demand for each SKU so business users understand the model
 Stockout risk heatmap — a store × product grid highlighting which combinations are most at risk given current policy
 Promo impact simulator — input a planned discount and see the predicted demand lift instantly
 Real data integration — connect to a public retail dataset (e.g. Kaggle's Rossmann or M5 competition data) to validate on real-world data


 🛠️ Tech Stack
ToolPurposeXGBoostDemand forecasting modelPandas / NumPyData processing & feature engineeringPlotlyInteractive chartsStreamlitDashboard frameworkDockerContainerized deploymentHugging Face SpacesCloud hosting
