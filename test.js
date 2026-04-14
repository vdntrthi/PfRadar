
        const INDIAN_TICKERS = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "INFY.NS", "ITC.NS", "HINDUNILVR.NS",
            "L&T.NS", "SBIN.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS", "TATAMOTORS.NS", "SUNPHARMA.NS",
            "TITAN.NS", "MARUTI.NS", "TATASTEEL.NS", "ULTRACEMCO.NS", "BAJAJFINSV.NS", "M&M.NS", "NTPC.NS", "AXISBANK.NS",
            "WIPRO.NS", "NESTLEIND.NS", "POWERGRID.NS", "ONGC.NS", "TECHM.NS", "HINDALCO.NS", "GRASIM.NS", "JSWSTEEL.NS",
            "ADANIENT.NS", "ADANIPORTS.NS", "CIPLA.NS", "HEROMOTOCO.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "EICHERMOT.NS",
            "DIVISLAB.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS", "LTIM.NS", "HDFCLIFE.NS", "COALINDIA.NS", "TATACONSUM.NS",
            "BPCL.NS", "INDUSINDBK.NS", "ZOMATO.NS", "JIOFIN.NS", "TRENT.NS", "HAL.NS", "IREDA.NS"
        ];

        let selectedPortfolio = [];

        document.addEventListener('DOMContentLoaded', () => {
            const dl = document.getElementById('tickersList');
            INDIAN_TICKERS.forEach(t => {
                const opt = document.createElement('option');
                opt.value = t;
                dl.appendChild(opt);
            });

            document.getElementById('tickerSelect').addEventListener('keypress', function (e) {
                if (e.key === 'Enter') {
                    addTicker();
                }
            });
        });

        function addTicker() {
            const selectEl = document.getElementById('tickerSelect');
            let ticker = selectEl.value.trim().toUpperCase();
            if (!ticker) return;

            if (!ticker.includes('.')) {
                ticker += ".NS";
            }

            if (selectedPortfolio.find(p => p.ticker === ticker)) {
                alert("Ticker already added.");
                return;
            }

            selectedPortfolio.push({ ticker, weight: 0 });
            redistributeWeights();
            renderTable();
            selectEl.value = '';
        }

        function removeTicker(ticker) {
            selectedPortfolio = selectedPortfolio.filter(p => p.ticker !== ticker);
            if (selectedPortfolio.length > 0) {
                redistributeWeights();
            }
            renderTable();
            if (selectedPortfolio.length === 0) {
                document.getElementById('resultsContainer').style.display = 'none';
            }
        }

        function redistributeWeights() {
            if (selectedPortfolio.length === 0) return;
            const evenWeight = (1.0 / selectedPortfolio.length).toFixed(3);
            selectedPortfolio.forEach(p => p.weight = parseFloat(evenWeight));
            const sum = selectedPortfolio.reduce((acc, p) => acc + p.weight, 0);
            if (Math.abs(sum - 1.0) > 0.001) {
                selectedPortfolio[selectedPortfolio.length - 1].weight += (1.0 - sum);
                selectedPortfolio[selectedPortfolio.length - 1].weight = parseFloat(selectedPortfolio[selectedPortfolio.length - 1].weight.toFixed(3));
            }
        }

        function updateWeight(ticker, newWeight) {
            const val = parseFloat(newWeight);
            const item = selectedPortfolio.find(p => p.ticker === ticker);
            if (item) {
                item.weight = isNaN(val) ? 0 : val;
            }
        }

        function renderTable() {
            const tbody = document.getElementById('tickersTableBody');
            tbody.innerHTML = '';
            
            if (selectedPortfolio.length === 0) {
                document.getElementById('portfolioControls').style.display = 'none';
                return;
            }

            document.getElementById('portfolioControls').style.display = 'block';

            selectedPortfolio.forEach(item => {
                const tr = document.createElement('tr');
                tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';
                tr.innerHTML = `
                    <td style="padding: 0.75rem; font-weight: 600;">${item.ticker}</td>
                    <td style="padding: 0.75rem;">
                        <input type="number" step="0.01" min="0" max="1" class="table-input" value="${item.weight}" onchange="updateWeight('${item.ticker}', this.value)">
                    </td>
                    <td style="padding: 0.75rem; text-align: right;">
                        <button onclick="removeTicker('${item.ticker}')" class="remove-btn">Remove</button>
                    </td>
                `;
                tbody.appendChild(tr);
            });
        }

        const userReturnEl = document.getElementById('userReturn');
        const userRiskEl = document.getElementById('userRisk');
        const optReturnEl = document.getElementById('optReturn');
        const optRiskEl = document.getElementById('optRisk');
        const optWeightsEl = document.getElementById('optWeights');
        const userCagrEl = document.getElementById('userCagr');
        const optCagrEl = document.getElementById('optCagr');
        const userCagrSubEl = document.getElementById('userCagrSub');
        const optCagrSubEl = document.getElementById('optCagrSub');
        const loader = document.getElementById('loader');
        const loaderAsset = document.getElementById('loaderAsset');
        const loaderFrontier = document.getElementById('loaderFrontier');
        const canvas = document.getElementById('portfolioChart');
        const assetCanvas = document.getElementById('assetChart');
        const frontierCanvas = document.getElementById('frontierChart');
        let currentChart = null;
        let currentAssetChart = null;
        let currentFrontierChart = null;

        function formatCagr(value) {
            if (value === null || value === undefined) return { text: '--', cls: '' };
            const pct = (value * 100).toFixed(2);
            const cls = value >= 0 ? 'cagr-positive' : 'cagr-negative';
            const sign = value >= 0 ? '+' : '';
            return { text: `${sign}${pct}%`, cls };
        }

        function handleFetchReport() {
            const sum = selectedPortfolio.reduce((acc, p) => acc + (parseFloat(p.weight) || 0), 0);
            const warningEl = document.getElementById('weightWarning');
            if (Math.abs(sum - 1.0) > 0.05) { 
                warningEl.innerText = `Total weight must equal exactly 1.0. Current: ${sum.toFixed(3)}`;
                warningEl.style.display = 'block';
                return;
            }
            warningEl.style.display = 'none';

            const tickers = selectedPortfolio.map(p => p.ticker).join(', ');
            const weights = selectedPortfolio.map(p => p.weight).join(', ');
            const risk = document.getElementById('riskInput').value;
            
            document.getElementById('resultsContainer').style.display = 'block';

            fetchReport(tickers, weights, risk);
        }

        function fetchReport(tickers, weights, risk) {
            loader.style.display = 'flex';
            loaderAsset.style.display = 'flex';
            loaderFrontier.style.display = 'flex';
            canvas.style.display = 'none';
            assetCanvas.style.display = 'none';
            frontierCanvas.style.display = 'none';

            // Reset metrics
            userReturnEl.innerText = '...';
            optReturnEl.innerText = '...';
            userCagrEl.innerText = '...';
            optCagrEl.innerText = '...';

            fetch(`/api/report?tickers=${encodeURIComponent(tickers)}&weights=${encodeURIComponent(weights)}&risk=${encodeURIComponent(risk)}`)
                .then(response => {
                    if (!response.ok) return response.json().then(err => { throw new Error(err.detail) });
                    return response.json();
                })
                .then(data => {
                    // Return / Risk cards
                    userReturnEl.innerText = (data.expected_return * 100).toFixed(2) + '%';
                    userRiskEl.innerText = 'Risk: ' + (data.volatility * 100).toFixed(2) + '%';

                    const optReturn = data.target_risk_expected_return !== null ? data.target_risk_expected_return : data.meta.max_sharpe_annual_return;
                    const optRiskVal = data.target_risk_volatility !== null ? data.target_risk_volatility : data.meta.max_sharpe_annual_volatility;
                    const optWeights = data.target_risk_portfolio !== null ? data.target_risk_portfolio : data.optimal_weights;

                    optReturnEl.innerText = (optReturn * 100).toFixed(2) + '%';
                    optRiskEl.innerText = 'Risk: ' + (optRiskVal * 100).toFixed(2) + '%';

                    let weightsStr = "";
                    for (const [sym, w] of Object.entries(optWeights)) {
                        weightsStr += `<div><strong>${sym}:</strong> ${(w * 100).toFixed(1)}%</div>`;
                    }
                    optWeightsEl.innerHTML = weightsStr;

                    // CAGR cards
                    const userCagrData = formatCagr(data.user_portfolio_cagr);
                    userCagrEl.innerText = userCagrData.text;
                    userCagrEl.className = `metric-value ${userCagrData.cls}`;

                    const optCagrData = formatCagr(data.optimal_portfolio_cagr);
                    optCagrEl.innerText = optCagrData.text;
                    optCagrEl.className = `metric-value ${optCagrData.cls}`;

                    // Efficient frontier
                    drawFrontierChart(data.efficient_frontier_data);

                    // Performance charts
                    drawChart(data.historical_chart_data);
                    drawAssetChart(data.historical_chart_data);
                })
                .catch(err => {
                    loader.innerHTML = `<span style="color: #ef4444; width:100%; text-align:center;">Error: ${err.message}</span>`;
                    loaderAsset.style.display = 'none';
                    loaderFrontier.style.display = 'none';
                });
        }

        function drawFrontierChart(frontierData) {
            if (!frontierData || !frontierData.cloud_volatilities) {
                loaderFrontier.style.display = 'flex';
                loaderFrontier.innerHTML = `<span style="color: #ef4444">No frontier data available</span>`;
                return;
            }

            loaderFrontier.style.display = 'none';
            frontierCanvas.style.display = 'block';

            if (currentFrontierChart) {
                currentFrontierChart.destroy();
            }

            const ctx = frontierCanvas.getContext('2d');

            // Build scatter data: random cloud
            const cloudData = frontierData.cloud_volatilities.map((v, i) => ({
                x: v * 100,
                y: frontierData.cloud_returns[i] * 100,
            }));

            // Key portfolio points
            const userPoint = [{
                x: frontierData.user_portfolio.volatility * 100,
                y: frontierData.user_portfolio.return * 100,
            }];
            const optPoint = [{
                x: frontierData.optimal_portfolio.volatility * 100,
                y: frontierData.optimal_portfolio.return * 100,
            }];
            const minVarPoint = [{
                x: frontierData.min_variance.volatility * 100,
                y: frontierData.min_variance.return * 100,
            }];
            const maxSharpePoint = [{
                x: frontierData.max_sharpe.volatility * 100,
                y: frontierData.max_sharpe.return * 100,
            }];

            currentFrontierChart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        {
                            label: 'Random Portfolios',
                            data: cloudData,
                            backgroundColor: 'rgba(148, 163, 184, 0.2)',
                            borderColor: 'rgba(148, 163, 184, 0.35)',
                            pointRadius: 2.5,
                            pointHoverRadius: 5,
                            order: 5,
                        },
                        {
                            label: 'Your Portfolio',
                            data: userPoint,
                            backgroundColor: '#38bdf8',
                            borderColor: '#fff',
                            borderWidth: 2,
                            pointRadius: 10,
                            pointHoverRadius: 14,
                            pointStyle: 'circle',
                            order: 1,
                        },
                        {
                            label: 'Optimized Portfolio',
                            data: optPoint,
                            backgroundColor: '#a855f7',
                            borderColor: '#fff',
                            borderWidth: 2,
                            pointRadius: 10,
                            pointHoverRadius: 14,
                            pointStyle: 'rectRounded',
                            order: 1,
                        },
                        {
                            label: 'Min Variance',
                            data: minVarPoint,
                            backgroundColor: '#22c55e',
                            borderColor: '#fff',
                            borderWidth: 2,
                            pointRadius: 8,
                            pointHoverRadius: 12,
                            pointStyle: 'star',
                            order: 2,
                        },
                        {
                            label: 'Max Sharpe',
                            data: maxSharpePoint,
                            backgroundColor: '#f59e0b',
                            borderColor: '#fff',
                            borderWidth: 2,
                            pointRadius: 8,
                            pointHoverRadius: 12,
                            pointStyle: 'triangle',
                            order: 2,
                        },
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'nearest',
                        intersect: true,
                    },
                    plugins: {
                        legend: {
                            display: false,
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleColor: '#94a3b8',
                            bodyColor: '#f8fafc',
                            borderColor: 'rgba(255,255,255,0.12)',
                            borderWidth: 1,
                            padding: 14,
                            cornerRadius: 10,
                            titleFont: { family: 'Inter', size: 13 },
                            bodyFont: { family: 'Inter', size: 14, weight: 600 },
                            callbacks: {
                                title: (items) => items[0]?.dataset.label || '',
                                label: (ctx) => `Return: ${ctx.parsed.y.toFixed(2)}%  |  Risk: ${ctx.parsed.x.toFixed(2)}%`,
                            }
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Annualized Volatility (%)',
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 13, weight: 500 },
                            },
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: {
                                color: '#64748b',
                                font: { family: 'Inter' },
                                callback: (val) => val.toFixed(1) + '%',
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Annualized Return (%)',
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 13, weight: 500 },
                            },
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: {
                                color: '#64748b',
                                font: { family: 'Inter' },
                                callback: (val) => val.toFixed(1) + '%',
                            },
                        }
                    }
                }
            });
        }

        function drawChart(chartData) {
            if (!chartData || !chartData.dates) {
                loader.style.display = 'flex';
                loader.innerHTML = `<span style="color: #ef4444">No historical chart data found</span>`;
                return;
            }

            loader.style.display = 'none';
            canvas.style.display = 'block';

            if (currentChart) {
                currentChart.destroy();
            }

            const toPercent = arr => arr.map(v => v * 100);
            const ctx = canvas.getContext('2d');

            const optGradient = ctx.createLinearGradient(0, 0, 0, 400);
            optGradient.addColorStop(0, 'rgba(168, 85, 247, 0.4)');
            optGradient.addColorStop(1, 'rgba(168, 85, 247, 0.0)');

            const userGradient = ctx.createLinearGradient(0, 0, 0, 400);
            userGradient.addColorStop(0, 'rgba(56, 189, 248, 0.3)');
            userGradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

            currentChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.dates,
                    datasets: [
                        {
                            label: 'Optimal Portfolio',
                            data: toPercent(chartData.optimal_portfolio),
                            borderColor: '#a855f7',
                            backgroundColor: optGradient,
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 6,
                        },
                        {
                            label: 'Your Portfolio',
                            data: toPercent(chartData.user_portfolio),
                            borderColor: '#38bdf8',
                            backgroundColor: userGradient,
                            borderWidth: 3,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 6,
                        },
                        {
                            label: 'Nifty 50 Benchmark',
                            data: toPercent(chartData.benchmark_nifty50),
                            borderColor: '#10b981',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 6,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#f8fafc',
                                font: { family: 'Inter', size: 13, weight: 500 },
                                usePointStyle: true,
                                boxWidth: 8
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.9)',
                            titleColor: '#94a3b8',
                            bodyColor: '#f8fafc',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            titleFont: { family: 'Inter', size: 13 },
                            bodyFont: { family: 'Inter', size: 14, weight: 600 },
                            callbacks: {
                                label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: { color: '#64748b', maxTicksLimit: 8, font: { family: 'Inter' } }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: {
                                color: '#64748b',
                                font: { family: 'Inter' },
                                callback: (val) => val + '%'
                            }
                        }
                    }
                }
            });
        }

        function drawAssetChart(chartData) {
            if (!chartData || !chartData.dates || !chartData.assets) {
                loaderAsset.style.display = 'flex';
                loaderAsset.innerHTML = `<span style="color: #ef4444">No asset chart data found</span>`;
                return;
            }

            loaderAsset.style.display = 'none';
            assetCanvas.style.display = 'block';

            if (currentAssetChart) {
                currentAssetChart.destroy();
            }

            const toPercent = arr => arr.map(v => v * 100);
            const ctx = assetCanvas.getContext('2d');

            const colors = ['#f43f5e', '#3b82f6', '#eab308', '#22c55e', '#a855f7', '#fb923c'];
            const datasets = [];
            let colorIdx = 0;

            for (const [sym, dataArr] of Object.entries(chartData.assets)) {
                datasets.push({
                    label: sym,
                    data: toPercent(dataArr),
                    borderColor: colors[colorIdx % colors.length],
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                });
                colorIdx++;
            }

            currentAssetChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: chartData.dates,
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                            labels: {
                                color: '#f8fafc',
                                font: { family: 'Inter', size: 13, weight: 500 },
                                usePointStyle: true,
                                boxWidth: 8
                            }
                        },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.9)',
                            titleColor: '#94a3b8',
                            bodyColor: '#f8fafc',
                            borderColor: 'rgba(255,255,255,0.1)',
                            borderWidth: 1,
                            padding: 12,
                            cornerRadius: 8,
                            titleFont: { family: 'Inter', size: 13 },
                            bodyFont: { family: 'Inter', size: 14, weight: 600 },
                            callbacks: {
                                label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(2)}%`
                            }
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: { color: '#64748b', maxTicksLimit: 8, font: { family: 'Inter' } }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false },
                            ticks: {
                                color: '#64748b',
                                font: { family: 'Inter' },
                                callback: (val) => val + '%'
                            }
                        }
                    }
                }
            });
        }
    