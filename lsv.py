
#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import numpy as np
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
from xgboost import XGBRegressor
import base64
from io import BytesIO
from PIL import Image

# 数据处理函数
def remove_lines_before_header(input_file, temp_file, header="Potential/V, Current/A"):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    start_index = next((i for i, line in enumerate(lines) if header in line), None)
    if start_index is not None:
        with open(temp_file, 'w') as file:
            file.writelines(lines[start_index:])
    else:
        raise ValueError(f"Header '{header}' not found in the file.")

def txt_to_excel_clean(input_txt, output_excel, input_file):
    data = pd.read_csv(input_txt, sep=",", skip_blank_lines=True)
    data.dropna(how="all", inplace=True)
    data.columns = data.columns.str.strip()
    if "Potential/V" in data.columns:
        if "HER" in input_file and "6M" in input_file:
            data["Modified Potential/V"] = (0.059 * 14.778 + 0.098 + data["Potential/V"])
        elif "HER" in input_file and "1M" in input_file:
            data["Modified Potential/V"] = (0.0592 * 14 + 0.098 + data["Potential/V"])
        elif "OER" in input_file and "6M" in input_file:
            data["Modified Potential/V"] = (0.059 * 14.778 + 0.098 + data["Potential/V"] - 1.23)
        elif "OER" in input_file and "1M" in input_file:
            data["Modified Potential/V"] = (0.0592 * 14 + 0.098 + data["Potential/V"] - 1.23)
    if "Current/A" in data.columns:
        data["Modified Current/mA"] = (1000 / 0.64 * data["Current/A"])
    data.to_excel(output_excel, index=False, engine="openpyxl")
    return pd.read_excel(output_excel)

# 自定义训练测试切分
def custom_train_test_split(x_data, y_data, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    indices = np.arange(x_data.shape[0])
    np.random.shuffle(indices)
    split_index = int(x_data.shape[0] * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return x_data[train_indices], x_data[test_indices], y_data[train_indices], y_data[test_indices]

app= Dash(__name__)
server = app.server
# 蓝白主题样式
app.layout = html.Div(
    [
        html.H1(
            "LSV 数据分析工具",
            style={"textAlign": "center", "color": "#0056b3", "marginBottom": "20px"},
        ),
        html.Div(
            [
                html.Label("上传 TXT 文件:", style={"fontWeight": "bold", "color": "#004080"}),
                dcc.Upload(
                    id="upload-data",
                    children=html.Div(["拖放文件或点击上传"]),
                    style={
                        "width": "100%",
                        "height": "60px",
                        "lineHeight": "60px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "5px",
                        "textAlign": "center",
                        "margin": "10px",
                        "backgroundColor": "#eaf4fc",
                        "color": "#0056b3",
                    },
                    multiple=True,
                ),
                html.Div(id="upload-status", style={"marginTop": "10px", "color": "#333"}),
                html.Label("是否启用拟合功能:", style={"fontWeight": "bold", "color": "#004080"}),
                dcc.RadioItems(
                    id="fit-curve",
                    options=[
                        {"label": "是", "value": "yes"},
                        {"label": "否", "value": "no"},
                    ],
                    value="no",
                    inline=True,
                    style={"marginBottom": "10px", "color": "#0056b3"},
                ),
                html.Label("目标 x 值 (可选):", style={"fontWeight": "bold", "color": "#004080"}),
                dcc.Input(
                    id="x-target",
                    type="number",
                    placeholder="输入目标 x 值",
                    style={
                        "width": "100%",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "border": "1px solid #ccc",
                        "marginBottom": "10px",
                    },
                ),
                html.Label("选择图表模板:", style={"fontWeight": "bold", "color": "#004080"}),
                dcc.Dropdown(
                    id="chart-template",
                    options=[
                        {"label": "白色背景", "value": "plotly_white"},
                        {"label": "经典", "value": "plotly"},
                        {"label": "深色背景", "value": "plotly_dark"},
                    ],
                    value="plotly_white",
                    style={"marginBottom": "10px"},
                ),
                html.Button(
                    "开始分析",
                    id="analyze-button",
                    style={
                        "margin": "10px",
                        "backgroundColor": "#0056b3",
                        "color": "white",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={
                "padding": "20px",
                "border": "1px solid #ccc",
                "borderRadius": "5px",
                "backgroundColor": "#f4faff",
            },
        ),
        html.Div(id="output-graph", style={"marginTop": "20px"}),
    ],
    style={"maxWidth": "800px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
)

# 回调函数：显示上传文件的名称和数量
@app.callback(
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
)
def update_upload_status(contents, filenames):
    if not contents or not filenames:
        return "未上传文件。"
    file_count = len(filenames)
    file_list = html.Ul([html.Li(file) for file in filenames])
    return html.Div(
        [
            html.P(f"已上传 {file_count} 个文件：", style={"fontWeight": "bold"}),
            file_list,
        ]
    )

# 回调函数逻辑与图表生成
@app.callback(
    Output("output-graph", "children"),
    Input("analyze-button", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("fit-curve", "value"),
    State("x-target", "value"),
    State("chart-template", "value"),
    prevent_initial_call=True,
)
def analyze_data(n_clicks, contents, filenames, fit_curve, x_target, chart_template):
    if not contents or not filenames:
        raise PreventUpdate

    temp_file = "temp_cleaned_data.txt"
    dfs = []

    for content, filename in zip(contents, filenames):
        content_type, content_string = content.split(",")
        decoded = base64.b64decode(content_string).decode("utf-8")
        with open("temp.txt", "w") as f:
            f.write(decoded)

        remove_lines_before_header("temp.txt", temp_file)
        if "OER" in filename:
            output_excel = "OER.xlsx"
        elif "HER" in filename:
            output_excel = "HER.xlsx"
        else:
            output_excel = "output_data.xlsx"
        df = txt_to_excel_clean(temp_file, output_excel, filename)
        df["Source"] = filename
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    fig = px.scatter(
        combined_df,
        x="Modified Potential/V",
        y="Modified Current/mA",
        color="Source",
        labels={"x": "Modified Potential (V)", "y": "Modified Current (mA)"},
        title="Comparison of Modified Potential vs. Modified Current",
        template=chart_template,
    )

    if fit_curve == "yes":
        x_data = combined_df["Modified Potential/V"].values.reshape(-1, 1)
        y_data = combined_df["Modified Current/mA"].values
        X_train, X_test, y_train, y_test = custom_train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
        model.fit(X_train, y_train)

        fit_x = np.linspace(min(x_data), max(x_data), 600).reshape(-1, 1)
        fit_y = model.predict(fit_x)

        fig.add_scatter(x=fit_x.flatten(), y=fit_y, mode="lines", name="拟合曲线")

        if x_target is not None:
            y_target = model.predict(np.array([[x_target]]))[0]
            fig.add_scatter(
                x=[x_target],
                y=[y_target],
                mode="markers",
                marker=dict(color="red", size=10),
                name=f"目标点 ({x_target}, {y_target:.4f})",
            )

    return dcc.Graph(figure=fig)

if __name__ == "__main__":
    app.run_server(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))



# In[ ]:




