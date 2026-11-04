const API = 'http://127.0.0.1:8000';

let chart = null;
let last = [];

const $ = id => document.getElementById(id);

const fmt = (v, d = 2) =>
    Number.isFinite(Number(v))
        ? Number(v).toLocaleString('en-IN', {
            maximumFractionDigits: d
        })
        : '—';

async function get(path) {
    const r = await fetch(API + path);

    if (!r.ok) {
        throw Error(r.status + ' ' + r.statusText);
    }

    return r.json();
}

// Update backend connection status
function state(ok) {
    $('api-status').textContent = ok
        ? 'Connected to FastAPI'
        : 'Could not reach localhost:8000';

    $('status').textContent = ok ? 'Online' : 'Offline';
    $('status').style.color = ok ? '#159a75' : '#c76c6c';
    $('dot').style.background = ok ? '#18b68a' : '#df7777';
}

// Find a value inside a nested object
function deep(obj, keys) {
    if (!obj || typeof obj !== 'object') return null;

    for (const k of keys) {
        if (
            obj[k] !== undefined &&
            (typeof obj[k] === 'number' ||
             typeof obj[k] === 'string')
        ) {
            return obj[k];
        }
    }

    for (const v of Object.values(obj)) {
        if (v && typeof v === 'object') {
            const x = deep(v, keys);

            if (x !== null) return x;
        }
    }

    return null;
}

// Load dataset information and backend model information
async function overview() {
    try {
        const [statusData, metadata] = await Promise.all([
            get('/api/v1/status').catch(() => ({})),
            get('/api/v1/metadata')
        ]);

        // Total dataset records
        if (metadata.rows !== undefined && $('total')) {
            $('total').textContent = fmt(metadata.rows, 0);
        }

        // Available forecasting models
        const availableModels = deep(statusData, [
            'available_models',
            'model_count'
        ]);

        if (availableModels !== null && $('models')) {
            $('models').textContent = Array.isArray(availableModels)
                ? fmt(availableModels.length, 0)
                : fmt(availableModels, 0);
        }

        // Optional dataset details
        if ($('date-range')) {
            $('date-range').textContent =
                `${metadata.date_min || '—'} to ${metadata.date_max || '—'}`;
        }

        if ($('store-count') && Array.isArray(metadata.stores)) {
            $('store-count').textContent =
                fmt(metadata.stores.length, 0);
        }

        if ($('item-count') && Array.isArray(metadata.items)) {
            $('item-count').textContent =
                fmt(metadata.items.length, 0);
        }

        $('updated').textContent =
            'Updated ' + new Date().toLocaleTimeString();

        console.log('Dataset metadata:', metadata);

    } catch (error) {
        console.error('Overview loading error:', error);
        $('updated').textContent = 'Metadata unavailable';
    }
}

// Escape HTML before inserting dynamic text
function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[c]));
}

// Load and display actual backend model metrics
async function metrics() {
    try {
        const data = await get('/api/v1/metrics');

        const periods = [
            {
                key: 'full_year',
                label: 'Full Year'
            },
            {
                key: '30_day',
                label: '30 Days'
            }
        ];

        let rows = '';

        periods.forEach(period => {
            const models = Array.isArray(data[period.key])
                ? data[period.key]
                : [];

            const validModels = models.filter(model =>
                Number.isFinite(Number(model.MAE)) &&
                Number.isFinite(Number(model.RMSE))
            );

            if (!validModels.length) return;

            const bestMAE = Math.min(
                ...validModels.map(model => Number(model.MAE))
            );

            const bestRMSE = Math.min(
                ...validModels.map(model => Number(model.RMSE))
            );

            validModels.forEach(model => {
                const isBest =
                    Number(model.MAE) === bestMAE &&
                    Number(model.RMSE) === bestRMSE;

                rows += `
                    <tr>
                        <td>
                            <b>
                                ${esc(period.label)} ·
                                ${esc(model.Model)}
                            </b>
                        </td>

                        <td>${fmt(model.MAE, 2)}</td>

                        <td>${fmt(model.RMSE, 2)}</td>

                        <td>
                            ${isBest ? 'Best in period' : 'Evaluated'}
                        </td>
                    </tr>
                `;
            });
        });

        $('metrics').innerHTML = rows ||
            '<tr><td colspan="4">No metric records available.</td></tr>';

    } catch (error) {
        console.error('Metrics loading error:', error);

        $('metrics').innerHTML =
            '<tr><td colspan="4">Could not load model metrics. Check the backend.</td></tr>';
    }
}

// Read forecast data returned by the backend
function parseForecast(d) {
    const a =
        d.forecasts ||
        d.forecast ||
        d.predictions ||
        d.results ||
        [];

    if (!Array.isArray(a)) return [];

    return a.map(x => ({
        date: x.date || x.ds || x.timestamp || '',

        value: Number(
            x.predicted_sales ??
            x.prediction ??
            x.forecast ??
            x.sales ??
            x.yhat
        )

    })).filter(x => Number.isFinite(x.value));
}

// Display forecast results and chart
function render(d, model, h) {
    const a = parseForecast(d);

    if (!a.length) {
        throw Error(
            'API responded, but no forecast array was found. Inspect the response JSON field names.'
        );
    }

    last = a;

    const total = a.reduce(
        (s, x) => s + x.value,
        0
    );

    const peak = a.reduce(
        (x, y) => x.value > y.value ? x : y
    );

    // Update forecast summary cards
    $('sum').textContent = fmt(total, 0);
    $('avg').textContent = fmt(total / a.length, 0);
    $('peak').textContent = fmt(peak.value, 0);

    $('result-sub').textContent =
        model + ' · ' + h + '-day forecast';

    $('result-status').textContent = 'FORECAST READY';
    $('result-status').style.color = '#159a75';

    $('empty').classList.add('hide');
    $('output').classList.remove('hide');

    // Populate daily forecast table
    $('rows').innerHTML = a.map(x => `
        <tr>
            <td>${esc(x.date || '—')}</td>
            <td>${fmt(x.value)}</td>
        </tr>
    `).join('');

    $('note').textContent =
        d.note ||
        'Forecast values are returned by your trained backend model. ARIMA/SARIMA forecasts are aggregate daily sales across all stores and items.';

    // Create forecast chart
    if (chart) {
        chart.destroy();
    }

    chart = new Chart($('chart'), {
        type: 'line',

        data: {
            labels: a.map(x => x.date),

            datasets: [{
                data: a.map(x => x.value),
                label: 'Predicted sales',
                borderColor: '#6078e8',
                backgroundColor: 'rgba(96,120,232,.12)',
                borderWidth: 2.5,
                fill: true,
                tension: 0.32,
                pointRadius: a.length > 20 ? 0 : 3
            }]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            plugins: {
                legend: {
                    display: false
                }
            },

            scales: {
                x: {
                    grid: {
                        display: false
                    },

                    ticks: {
                        color: '#9aa4b4',
                        maxTicksLimit: 7,
                        maxRotation: 0
                    }
                },

                y: {
                    grid: {
                        color: '#eff2f7'
                    },

                    ticks: {
                        color: '#9aa4b4'
                    }
                }
            }
        }
    });
}

// Handle forecast form submission
$('form').addEventListener('submit', async e => {
    e.preventDefault();

    const model = $('model').value;

    const selectedHorizon =
        document.querySelector('[name=horizon]:checked');

    if (!selectedHorizon) {
        $('error').textContent =
            'Please select a forecast horizon.';

        $('error').classList.remove('hide');
        return;
    }

    const h = Number(selectedHorizon.value);
    const btn = $('generate');

    $('error').classList.add('hide');

    btn.disabled = true;
    btn.textContent = 'Generating forecast…';

    $('result-status').textContent = 'PROCESSING';

    try {
        let path;
        let payload;

        if (model === 'random_forest') {

            path = '/api/v1/forecast/random-forest';

            payload = {
                horizon: h
            };

            if ($('store').value) {
                payload.store = Number($('store').value);
            }

            if ($('item').value) {
                payload.item = Number($('item').value);
            }

        } else {

            path = '/api/v1/forecast/time-series';

            payload = {
                model,
                horizon: h
            };
        }

        const r = await fetch(API + path, {
            method: 'POST',

            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },

            body: JSON.stringify(payload)
        });

        const d = await r.json().catch(() => ({}));

        if (!r.ok) {
            throw Error(
                d.detail
                    ? (
                        typeof d.detail === 'string'
                            ? d.detail
                            : JSON.stringify(d.detail)
                    )
                    : r.status + ' ' + r.statusText
            );
        }

        render(
            d,
            model === 'random_forest'
                ? 'Random Forest'
                : model.toUpperCase(),
            h
        );

    } catch (err) {

        $('result-status').textContent = 'REQUEST FAILED';

        $('error').textContent =
            'Forecast could not be generated: ' +
            err.message +
            ' Verify that the backend is running and the request fields match its schema.';

        $('error').classList.remove('hide');

    } finally {

        btn.disabled = false;

        btn.innerHTML =
            'Generate forecast <b>↗</b>';
    }
});

// Update model description and Random Forest inputs
$('model').addEventListener('change', () => {
    const m = $('model').value;

    $('rf-fields').classList.toggle(
        'hide',
        m !== 'random_forest'
    );

    $('model-note').textContent = {
        random_forest:
            'Machine learning model using lag and rolling demand features.',

        arima:
            'Time-series model for patterns in historical aggregate sales.',

        sarima:
            'Seasonal time-series model designed for recurring demand patterns.'

    }[m];
});

// Retry backend connection
$('retry').addEventListener('click', connect);

// Refresh model metrics
$('refresh').addEventListener('click', metrics);

// Export forecast results as CSV
$('csv').addEventListener('click', () => {
    if (!last.length) return;

    const csv = [
        ['date', 'predicted_sales'],

        ...last.map(x => [
            x.date,
            x.value
        ])

    ].map(r =>
        r.map(v =>
            '"' + String(v).replaceAll('"', '""') + '"'
        ).join(',')
    ).join('\n');

    const url = URL.createObjectURL(
        new Blob([csv], {
            type: 'text/csv'
        })
    );

    const a = document.createElement('a');

    a.href = url;
    a.download = 'retailpulse_forecast.csv';

    a.click();

    URL.revokeObjectURL(url);
});

// Check backend connection and load dashboard data
async function connect() {
    try {
        await get('/health');

        state(true);

        await Promise.all([
            overview(),
            metrics()
        ]);

    } catch (e) {
        state(false);
    }
}

// Initial dashboard loading
connect();