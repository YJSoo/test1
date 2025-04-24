from flask import Flask, request, jsonify, render_template,Response
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# 加载数据
df = pd.read_excel("yearwater_converted.xlsx")
df_sun = pd.read_csv("sunyear_processed.csv")
df_price = pd.read_excel("yearprice.xlsx")
for col in df_price.columns[1:]:  # 跳过第一列"农产品名称"
    df_price[col] = pd.to_numeric(df_price[col], errors='coerce')  # 转换失败设为NaN


# 获取所有地区
regions = df["PR"].dropna().unique().tolist()
products = df_price["农产品名称"].dropna().unique().tolist()

@app.route('/')
def index():
    return render_template("index.html", regions=regions)

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    try:
        data = request.get_json(silent=True)
        region = data.get('region') if data else request.form.get('region')
        year = data.get('year') if data else request.form.get('year')

        if not region or not year:
            return render_template("results.html", error="缺少必要参数: 地区或年份")

        try:
            year = int(year)
        except ValueError:
            return render_template("results.html", error="年份必须是整数")

        # ARIMA预测函数
        def arima_forecast(series, steps=1):
            try:
                model = ARIMA(series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=steps)
                return forecast[0] if steps == 1 else forecast.tolist()
            except Exception as e:
                print(f"ARIMA预测错误: {str(e)}")
                return np.nan

        # 获取地区数据
        target_row = df[df["PR"] == region]
        if len(target_row) == 0:
            return render_template("results.html", error=f"未找到地区 {region}")

        if year >= 2025:
            # 预测未来数据
            history = target_row[[str(y) for y in range(1950, 2023) if str(y) in df.columns]].values[0]
            history = pd.Series(history).dropna()

            if len(history) < 10:
                return render_template("results.html", error="历史数据不足，无法进行预测")

            steps = year - 2022
            forecast = arima_forecast(history, steps=steps)

            if year == 2025:
                rain = forecast[0] if isinstance(forecast, list) else forecast
            elif year == 2026:
                rain = forecast[1] if isinstance(forecast, list) and len(forecast) > 1 else np.nan
            else:
                return render_template("results.html", error="只支持预测2025或2026年")
        else:
            # 获取历史数据
            if str(year) not in df.columns:
                return render_template("results.html", error=f"无{year}年数据")

            rain = target_row[str(year)].values[0]

        target_row_sun = df_sun[df_sun['Region'] == region]

        if year >= 2025:
            # Forecast future sunshine data
            history_sun = target_row_sun[
                (target_row_sun['Year'] >= 1960) & (target_row_sun['Year'] <= 2022)
                ].sort_values('Year')['Sunshine_Hours']

            # 确保数据是Series格式且去除缺失值
            history_sun = pd.Series(history_sun).dropna()

            if len(history_sun) < 10:
                return render_template("results.html", error="日照数据历史数据不足，无法进行预测")

            steps_sun = year - 2022
            forecast_sun = arima_forecast(history_sun, steps=steps_sun)

            if year == 2025:
                sunshine = forecast_sun[0] if isinstance(forecast_sun, list) else forecast_sun
            elif year == 2026:
                sunshine = forecast_sun[1] if isinstance(forecast_sun, list) and len(forecast_sun) > 1 else np.nan
            else:
                return render_template("results.html", error="只支持预测2025或2026年")
        else:
            # Get historical sunshine data
            if str(year) not in df_sun.columns:
                return render_template("results.html", error=f"无{year}年日照数据")

            sunshine = target_row_sun[str(year)].values[0]/365

        product_results = []
        if year >= 2025:
            for product in products:
                row = df_price[df_price["农产品名称"] == product]
                if len(row) == 0:
                    continue

                price_columns = [y for y in range(2005, 2024) if y in df_price.columns]
                history_price = row[price_columns].values[0]
                history_price = pd.Series(history_price).dropna()

                if len(history_price) < 2:
                    continue

                last_year_price = history_price.iloc[-1]
                if len(history_price) < 10:
                    # 少于10个年份，用平均值
                    predicted_price = np.mean(history_price)
                else:
                    steps_price = year - 2023
                    forecast_price = arima_forecast(history_price, steps=steps_price)
                    predicted_price = forecast_price[0] if isinstance(forecast_price,
                                                                      (np.ndarray, list)) else forecast_price

                if pd.isna(predicted_price) or last_year_price == 0:
                    continue

                growth_rate = (predicted_price - last_year_price) / last_year_price
                if growth_rate > -0.05:
                    product_results.append({
                        "product": product,
                        "predicted_price": round(float(predicted_price), 2),
                        "growth_rate": round(float(growth_rate) * 100, 2)
                    })
        else:
            # 2024及之前的数据直接读取原始数据
            for product in products:
                row = df_price[df_price["农产品名称"] == product]
                if len(row) == 0 or str(year) not in df_price.columns:
                    continue
                price = row[str(year)].values[0]
                if not pd.isna(price):
                    product_results.append({
                        "product": product,
                        "predicted_price": round(float(price), 2),
                        "growth_rate": None
                    })

        # Return the result
        return {
            'rainfall': round(float(rain), 2),
            'sunshine': round(float(sunshine), 2),
            'price_result': product_results
        }



    # return render_template("results.html", region=region, year=year, rain=rain, predicted=(year >= 2025))

    except Exception as e:
        return Response(f"错误: {str(e)}", mimetype='text/plain')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
