# Stock-technical-analysis
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from finta import TA
import mplfinance as mpf
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import webbrowser
from scipy.signal import find_peaks
filename = 'data-Fintech2023.xlsx'
data = pd.read_excel(filename, sheet_name='Price', engine='openpyxl')


# Sử dụng stack để chuyển các cột thành hàng
data2 = data.T
data2

# Đặt dòng đầu tiên làm header mới
new_header = data2.iloc[0]
df = data2[1:]
df.columns = new_header

# Reset index
df = df.reset_index(drop=True)

# Sử dụng stack để chuyển các cột thành hàng
data2 = df.T

while True:
    # Nhập mã cổ phiếu từ người dùng
    ma_co_phieu = input("Nhập mã cổ phiếu: ")

    # Tìm kiếm theo chỉ mã
    thong_tin_co_phieu = df[df['Code'].str.contains(ma_co_phieu, case=False, regex=False)]

    # Tạo DataFrame mới từ thông tin của mã cổ phiếu
    df2 = thong_tin_co_phieu.copy()

    # Hiển thị thông tin về mã cổ phiếu
    if not thong_tin_co_phieu.empty:
        print("Thông tin về mã cổ phiếu {}: \n{}".format(ma_co_phieu, df2))
    else:
        print("Không tìm thấy thông tin cho mã cổ phiếu:", ma_co_phieu) 
    df2 = df2.T
    # Sử dụng reset_index để chuyển index thành các cột mới
    df_moi_final = df2.reset_index()
    # Hiển thị DataFrame mới
    df_moi_final

    # Đặt tên mới cho cột
    df_moi_final.columns = ['Date','Close']
    df_moi_final = df_moi_final.iloc[1:]
    # Loại bỏ các dòng có ít nhất một giá trị NaN
    df_moi_final = df_moi_final.dropna()
    # Đọc dữ liệu từ DataFrame
    data = df_moi_final
    # Assuming you've loaded your data into df_moi_final
    data = df_moi_final.reset_index(drop=True)  # Resetting the index for ease of reference

    # Identify MACD divergences
    def find_macd_divergences(data):
        bullish_lines = []
        bearish_lines = []

        for i in range(2, len(data)):
            if data['Close'][i] < data['Close'][i-2] and data['macd'][i] > data['macd'][i-2]:
                bullish_lines.append(([data['Date'][i-2], data['Date'][i]], [data['Close'][i-2], data['Close'][i]]))

            if data['Close'][i] > data['Close'][i-2] and data['macd'][i] < data['macd'][i-2]:
                bearish_lines.append(([data['Date'][i-2], data['Date'][i]], [data['Close'][i-2], data['Close'][i]]))

        return bullish_lines, bearish_lines

    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.Label('Chọn đường chỉ báo:'),
        dcc.Dropdown(
            id='indicator-dropdown',
            options=[
                {'label': 'Close and Moving Average (MA)', 'value': 'ma'},
                {'label': 'Bollinger Bands', 'value': 'bollinger'},
                {'label': 'MACD', 'value': 'macd'},
                {'label': 'RSI', 'value': 'rsi'},
                {'label': 'MACD with Divergence', 'value': 'macdwd'},
                {'label': 'RSI with Divergence', 'value': 'rsiwd'},
                {'label': 'Stochastic Oscillator', 'value': 'stochastic'}

            ],
            value='ma'
        ),
        dcc.Graph(id='technical-indicator-graph')
    ])

    @app.callback(
        Output('technical-indicator-graph', 'figure'),
        [Input('indicator-dropdown', 'value')]
    )
    def update_graph(selected_indicator):
        traces = []
        if selected_indicator == 'ma':
            trace1 = go.Scatter(x=data['Date'], y=data['Close'], name='Close')
            data['MA'] = data['Close'].rolling(window=20).mean()
            trace2 = go.Scatter(x=data['Date'], y=data['MA'], name='Moving Average')
            figure = {
                'data': [trace1, trace2],
                'layout': go.Layout(title='Close and Moving Average (MA)')
            }
        elif selected_indicator == 'bollinger':
            # Tính toán Bollinger Bands
            window = 20
            data['Bollinger_Middle'] = data['Close'].rolling(window=window).mean()
            data['Bollinger_Std'] = data['Close'].rolling(window=window).std()
            data['Bollinger_Upper'] = data['Bollinger_Middle'] + 2 * data['Bollinger_Std']
            data['Bollinger_Lower'] = data['Bollinger_Middle'] - 2 * data['Bollinger_Std']

            trace1 = go.Scatter(x=data['Date'], y=data['Close'], name='Close')
            trace2 = go.Scatter(x=data['Date'], y=data['Bollinger_Upper'], name='Bollinger Upper')
            trace3 = go.Scatter(x=data['Date'], y=data['Bollinger_Lower'], name='Bollinger Lower')

            figure = {
                'data': [trace1, trace2, trace3],
                'layout': go.Layout(title='Bollinger Bands')
            }

        elif selected_indicator == 'macdwd':
            # MACD Calculation Code
            short_window = 12
            long_window = 26
            data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
            data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
            data['macd'] = data['Short_MA'] - data['Long_MA']
            data['signal_line'] = data['macd'].rolling(window=9).mean()

            trace1 = go.Scatter(x=data['Date'], y=data['Close'], name='Close')
            trace2 = go.Scatter(x=data['Date'], y=data['macd'], name='MACD')
            trace3 = go.Scatter(x=data['Date'], y=data['signal_line'], name='Signal Line')

            # Divergence logic (add this to your existing code)
            macd_tops, _ = find_peaks(data['macd'], distance=10)
            macd_bottoms, _ = find_peaks(-data['macd'], distance=10)

            for i in range(1, len(macd_tops)):
                if data['Close'].iloc[macd_tops[i]] < data['Close'].iloc[macd_tops[i-1]]:
                    traces.append(go.Scatter(
                        x=[data['Date'].iloc[macd_tops[i]], data['Date'].iloc[macd_tops[i-1]]],
                        y=[data['macd'].iloc[macd_tops[i]], data['macd'].iloc[macd_tops[i-1]]],
                        mode='lines',
                        line=dict(color='red'),
                        name='Bearish Divergence'
                    ))

            for i in range(1, len(macd_bottoms)):
                if data['Close'].iloc[macd_bottoms[i]] > data['Close'].iloc[macd_bottoms[i-1]]:
                    traces.append(go.Scatter(
                        x=[data['Date'].iloc[macd_bottoms[i]], data['Date'].iloc[macd_bottoms[i-1]]],
                        y=[data['macd'].iloc[macd_bottoms[i]], data['macd'].iloc[macd_bottoms[i-1]]],
                        mode='lines',
                        line=dict(color='green'),
                        name='Bullish Divergence'
                    ))

            figure = {
                'data': [trace1, trace2, trace3] + traces,
                'layout': go.Layout(
                    title='MACD with Divergences',
                    height=700,  # Adjusted size
                    width=1400   # Adjusted size
                )
            }

        elif selected_indicator == 'rsiwd': 
            # ... RSI Calculation Code ...

            window = 14
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            tops, _ = find_peaks(rsi, distance=10, height=70)
            bottoms, _ = find_peaks(-rsi, distance=10, height=-30)

            for i in range(1, len(tops)):
                if data['Close'].iloc[tops[i]] < data['Close'].iloc[tops[i-1]]:
                    traces.append(go.Scatter(
                        x=[data['Date'].iloc[tops[i]], data['Date'].iloc[tops[i-1]]],
                        y=[rsi.iloc[tops[i]], rsi.iloc[tops[i-1]]],
                        mode='lines+markers',
                        line=dict(color='red'),
                        marker=dict(symbol='circle', size=8),
                        name='Bearish Divergence'
                    ))

            for i in range(1, len(bottoms)):
                if data['Close'].iloc[bottoms[i]] > data['Close'].iloc[bottoms[i-1]]:
                    traces.append(go.Scatter(
                        x=[data['Date'].iloc[bottoms[i]], data['Date'].iloc[bottoms[i-1]]],
                        y=[rsi.iloc[bottoms[i]], rsi.iloc[bottoms[i-1]]],
                        mode='lines+markers',
                        line=dict(color='green'),
                        marker=dict(symbol='circle', size=8),
                        name='Bullish Divergence'
                    ))

            trace_rsi = go.Scatter(x=data['Date'], y=rsi, name='RSI')
            traces.append(trace_rsi)

            traces.append(go.Scatter(x=data['Date'], y=[30] * len(data), name='Lower Boundary (30)', line=dict(dash='dash')))
            traces.append(go.Scatter(x=data['Date'], y=[70] * len(data), name='Upper Boundary (70)', line=dict(dash='dash')))

            figure = {
                'data': traces,
                'layout': go.Layout(title='RSI with Divergences')
            }
        elif selected_indicator == 'macd':
            # Tính toán MACD
            short_window = 12
            long_window = 26
            data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
            data['Long_MA'] = data['Close'].rolling(window=long_window).mean()
            data['macd'] = data['Short_MA'] - data['Long_MA']
            data['signal_line'] = data['macd'].rolling(window=9).mean()

            trace1 = go.Scatter(x=data['Date'], y=data['Close'], name='Close')
            trace2 = go.Scatter(x=data['Date'], y=data['macd'], name='MACD')
            trace3 = go.Scatter(x=data['Date'], y=data['signal_line'], name='Signal Line')

            figure = {
                'data': [trace1, trace2, trace3],
                'layout': go.Layout(title='MACD')
            }
        elif selected_indicator == 'rsi':    
             # Tính toán RSI
            window = 14
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            trace1 = go.Scatter(x=data['Date'], y=rsi, name='RSI')
            trace2 = go.Scatter(x=data['Date'], y=[30] * len(data), name='Lower Boundary (30)', line=dict(dash='dash'))
            trace3 = go.Scatter(x=data['Date'], y=[70] * len(data), name='Upper Boundary (70)', line=dict(dash='dash'))

            figure = {
                'data': [trace1, trace2, trace3],
                'layout': go.Layout(title='RSI (with 20 and 80 boundaries)')
            }
        elif selected_indicator == 'stochastic':
            # Tính toán Stochastic Oscillator chỉ dựa vào giá đóng cửa (Close)
            window = 14
            data['Close_Min'] = data['Close'].rolling(window=window).min()
            data['Close_Max'] = data['Close'].rolling(window=window).max()
            data['Stoch_K'] = 100 * (data['Close'] - data['Close_Min']) / (data['Close_Max'] - data['Close_Min'])
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()

            trace1 = go.Scatter(x=data['Date'], y=data['Stoch_K'], name='Stochastic K', line=dict(width=0.5))  # Điều chỉnh độ rộng của đường
            trace2 = go.Scatter(x=data['Date'], y=data['Stoch_D'], name='Stochastic D', line=dict(width=0.5))  # Điều chỉnh độ rộng của đường

            # Đường giới hạn 20 và 80
            trace3 = go.Scatter(x=data['Date'], y=[20] * len(data), name='Lower Boundary (20)', line=dict(dash='dash', width=1))
            trace4 = go.Scatter(x=data['Date'], y=[80] * len(data), name='Upper Boundary (80)', line=dict(dash='dash', width=1))

            figure = {
                'data': [trace1, trace2, trace3, trace4],  # Bao gồm cả đường giới hạn
                'layout': go.Layout(
                    title='Stochastic Oscillator (Close)',
                    margin=dict(l=100, r=120, t=30, b=40),  # Điều chỉnh biên
                    legend=dict(x=1.05, y=1),  # Vị trí bảng kí hiệu
                    yaxis=dict(range=[0, 100], dtick=20),  # Điều chỉnh phạm vi và bước của trục y
                    xaxis=dict(title='Date'),
                    showlegend=True
                )
            }

        return figure

    if __name__ == '__main__':
       app.run_server(debug=True, port = 8055)
       webbrowser.open_new("http://localhost:8055")
# At the end of all processing, ask if the user wants to continue
    cont = input("Bạn muốn tiếp tục không? (yes/no): ").lower()
    if cont != 'yes':
        break
