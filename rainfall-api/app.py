from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# 加载数据
df = pd.read_excel("yearwater_converted.xlsx")


@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    try:
        # 获取输入参数
        data = request.json
        region = data.get('region')
        year = data.get('year')

        # 验证参数
        if not region or not year:
            return jsonify({"error": "缺少必要参数: region或year"}), 400

        try:
            year = int(year)
        except ValueError:
            return jsonify({"error": "year必须是整数"}), 400

        # 定义ARIMA预测函数
        def arima_forecast(series, steps=1):
            try:
                model = ARIMA(series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=steps)
                return forecast[0] if steps == 1 else forecast.tolist()
            except Exception as e:
                print(f"ARIMA预测错误: {str(e)}")
                return np.nan

        # 处理请求
        if year >= 2025:
            # 预测未来数据
            target_row = df[df["PR"] == region]
            if len(target_row) == 0:
                return jsonify({
                    "error": f"未找到地区 {region}",
                    "available_regions": df["PR"].tolist()
                }), 404

            history = target_row[[str(y) for y in range(1950, 2023) if str(y) in df.columns]].values[0]
            history = pd.Series(history).dropna()

            if len(history) < 10:
                return jsonify({
                    "error": "历史数据不足，无法预测",
                    "available_years": [y for y in range(1950, 2023) if not pd.isna(history.get(str(y), np.nan))]
                }), 400

            # 预测指定年份
            years_to_predict = year - 2022
            forecast = arima_forecast(history, steps=years_to_predict)

            if year == 2025:
                rain = forecast[0] if isinstance(forecast, list) else forecast
            elif year == 2026:
                rain = forecast[1] if isinstance(forecast, list) and len(forecast) > 1 else np.nan
            else:
                return jsonify({
                    "error": "只能预测2025或2026年",
                    "max_predict_year": 2026
                }), 400
        else:
            # 获取历史数据
            target_row = df[df["PR"] == region]
            if len(target_row) == 0:
                return jsonify({
                    "error": f"未找到地区 {region}",
                    "available_regions": df["PR"].tolist()
                }), 404

            if str(year) not in df.columns:
                return jsonify({
                    "error": f"无{year}年数据",
                    "available_years": [col for col in df.columns if col.isdigit() and 1950 <= int(col) <= 2022]
                }), 404

            rain = target_row[str(year)].values[0]

        # 准备响应
        response = {
            "region": region,
            "year": year,
            "rain": float(rain) if not pd.isna(rain) else None,
            "unit": "mm×1000",
            "predicted": year >= 2025,
            "status": "success"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": f"服务器内部错误: {str(e)}",
            "status": "error"
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)