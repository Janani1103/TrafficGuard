// Crimson Analytics - Traffic Accident Severity Prediction
// Main Application JavaScript

// Global variables
let predictionHistory = [];
let charts = {};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    setupEventListeners();
});

// Launch app from splash screen
function launchApp() {
    const splashScreen = document.getElementById('splash-screen');
    const mainApp = document.getElementById('main-app');
    
    splashScreen.classList.add('hidden');
    setTimeout(() => {
        splashScreen.style.display = 'none';
        mainApp.classList.add('active');
    }, 500);
}

// Page navigation
function showPage(pageName) {
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Remove active class from all nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Show selected page
    document.getElementById(`${pageName}-page`).classList.add('active');
    
    // Add active class to clicked nav link
    event.target.classList.add('active');
    
    // Initialize charts for specific pages
    if (pageName === 'reports') {
        initializeReportCharts();
    }
    
    // Load visualizations page content
    if (pageName === 'visualizations') {
        loadVisualizations();
    }
}

// Setup event listeners
function setupEventListeners() {
    // Prediction form submission
    document.getElementById('prediction-form').addEventListener('submit', handlePrediction);
}

// Handle prediction
async function handlePrediction(event) {
    event.preventDefault();
    
    // Show loading overlay
    const loadingOverlay = document.getElementById('loading-overlay');
    loadingOverlay.classList.add('active');
    
    // Gather form data
    const formData = {
        location: parseInt(document.getElementById('location').value),
        weather: parseInt(document.getElementById('weather').value),
        road_type: parseInt(document.getElementById('road_type').value),
        vehicle_type: parseInt(document.getElementById('vehicle_type').value),
        driver_age: parseInt(document.getElementById('driver_age').value),
        casualties: parseInt(document.getElementById('casualties').value),
        speed_limit: parseInt(document.getElementById('speed_limit').value),
        time: parseInt(document.getElementById('time').value)
    };
    
    try {
        // Make API call
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPredictionResults(result);
            addToHistory(formData, result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please try again.');
    } finally {
        // Hide loading overlay
        loadingOverlay.classList.remove('active');
    }
}

// Display prediction results
function displayPredictionResults(result) {
    const resultCard = document.getElementById('result-card');
    resultCard.style.display = 'block';
    
    // Update severity badge
    const severityBadge = document.getElementById('severity-badge');
    severityBadge.textContent = result.prediction;
    severityBadge.className = 'severity-badge';
    
    if (result.prediction === 'Fatal') {
        severityBadge.classList.add('fatal');
    } else if (result.prediction === 'Serious Injury') {
        severityBadge.classList.add('serious');
    } else {
        severityBadge.classList.add('minor');
    }
    
    // Update confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    document.getElementById('confidence-fill').style.width = confidencePercent + '%';
    document.getElementById('confidence-value').textContent = confidencePercent + '%';
    
    // Update probability bars
    const probabilities = result.probabilities;
    updateProbabilityBar('fatal', probabilities['Fatal']);
    updateProbabilityBar('serious', probabilities['Serious Injury']);
    updateProbabilityBar('minor', probabilities['Minor Injury']);
    
    // Update risk factors
    const riskFactorsList = document.getElementById('risk-factors-list');
    if (result.risk_factors && result.risk_factors.length > 0) {
        riskFactorsList.innerHTML = result.risk_factors
            .map(factor => `<span class="risk-factor">⚠️ ${factor}</span>`)
            .join('');
    } else {
        riskFactorsList.innerHTML = '<span class="risk-factor">No significant risk factors</span>';
    }
    
    // Scroll to results
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Update probability bar
function updateProbabilityBar(type, probability) {
    const percent = (probability * 100).toFixed(1);
    document.getElementById(`prob-${type}`).textContent = percent + '%';
    document.getElementById(`prob-fill-${type}`).style.width = percent + '%';
}

// Add to prediction history
function addToHistory(formData, result) {
    const locationLabels = ['Urban', 'Suburban', 'Rural', 'Highway'];
    const weatherLabels = ['Clear', 'Rain', 'Snow', 'Fog'];
    
    const historyEntry = {
        id: predictionHistory.length + 1,
        location: locationLabels[formData.location],
        weather: weatherLabels[formData.weather],
        age: formData.driver_age,
        speed: formData.speed_limit,
        severity: result.prediction,
        confidence: (result.confidence * 100).toFixed(1) + '%'
    };
    
    predictionHistory.unshift(historyEntry);
    
    // Keep only last 10 entries
    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }
    
    updateHistoryTable();
}

// Update history table
function updateHistoryTable() {
    const tableBody = document.getElementById('history-table-body');
    
    if (predictionHistory.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-grey);">No predictions yet</td></tr>';
        return;
    }
    
    tableBody.innerHTML = predictionHistory.map(entry => `
        <tr>
            <td>${entry.id}</td>
            <td>${entry.location}</td>
            <td>${entry.weather}</td>
            <td>${entry.age}</td>
            <td>${entry.speed}</td>
            <td><span class="severity-badge ${entry.severity.toLowerCase().replace(' ', '-')}">${entry.severity}</span></td>
            <td>${entry.confidence}</td>
        </tr>
    `).join('');
}

// Initialize charts
function initializeCharts() {
    // Weather chart
    const weatherCtx = document.getElementById('weatherChart').getContext('2d');
    charts.weather = new Chart(weatherCtx, {
        type: 'bar',
        data: {
            labels: ['Clear', 'Rain', 'Snow', 'Fog'],
            datasets: [
                {
                    label: 'Fatal',
                    data: [770, 770, 770, 772],
                    backgroundColor: '#e74c3c'
                },
                {
                    label: 'Serious',
                    data: [760, 760, 760, 747],
                    backgroundColor: '#f39c12'
                },
                {
                    label: 'Minor',
                    data: [750, 750, 750, 772],
                    backgroundColor: '#27ae60'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 11 },
                        boxWidth: 12,
                        padding: 10
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { font: { size: 10 } },
                    grid: {
                        display: true,
                        drawBorder: false
                    }
                },
                x: {
                    ticks: { font: { size: 10 } },
                    grid: {
                        display: false
                    }
                }
            },
            layout: {
                padding: {
                    top: 10,
                    bottom: 10
                }
            }
        }
    });

    // Road type chart
    const roadCtx = document.getElementById('roadChart').getContext('2d');
    charts.road = new Chart(roadCtx, {
        type: 'bar',
        data: {
            labels: ['Highway', 'Intersection', 'Rural', 'Urban'],
            datasets: [
                {
                    label: 'Fatal',
                    data: [770, 770, 770, 772],
                    backgroundColor: '#e74c3c'
                },
                {
                    label: 'Serious',
                    data: [760, 760, 760, 747],
                    backgroundColor: '#f39c12'
                },
                {
                    label: 'Minor',
                    data: [750, 750, 750, 772],
                    backgroundColor: '#27ae60'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 11 },
                        boxWidth: 12,
                        padding: 10
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: { font: { size: 10 } },
                    grid: {
                        display: true,
                        drawBorder: false
                    }
                },
                x: {
                    ticks: { font: { size: 10 } },
                    grid: {
                        display: false
                    }
                }
            },
            layout: {
                padding: {
                    top: 10,
                    bottom: 10
                }
            }
        }
    });
}

// Initialize report charts
function initializeReportCharts() {
    // Vehicle distribution chart
    const vehicleCtx = document.getElementById('vehicleDistChart');
    if (vehicleCtx && !charts.vehicle) {
        charts.vehicle = new Chart(vehicleCtx.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Motorcycle', 'Car', 'Truck', 'Bus'],
                datasets: [{
                    data: [2283, 2283, 2283, 2282],
                    backgroundColor: ['#800000', '#c0392b', '#e74c3c', '#f39c12']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 11 },
                            boxWidth: 12,
                            padding: 10
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }

    // Severity distribution chart
    const severityCtx = document.getElementById('severityDistChart');
    if (severityCtx && !charts.severityDist) {
        charts.severityDist = new Chart(severityCtx.getContext('2d'), {
            type: 'pie',
            data: {
                labels: ['Fatal', 'Serious', 'Minor'],
                datasets: [{
                    data: [3082, 3027, 3022],
                    backgroundColor: ['#e74c3c', '#f39c12', '#27ae60']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { size: 11 },
                            boxWidth: 12,
                            padding: 10
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }

    // Weather impact chart
    const weatherImpactCtx = document.getElementById('weatherImpactChart');
    if (weatherImpactCtx && !charts.weatherImpact) {
        charts.weatherImpact = new Chart(weatherImpactCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Clear', 'Rain', 'Snow', 'Fog'],
                datasets: [{
                    label: 'Accident Count',
                    data: [2283, 2283, 2283, 2282],
                    backgroundColor: '#800000'
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
                    y: {
                        beginAtZero: true,
                        ticks: { font: { size: 10 } },
                        grid: {
                            display: true,
                            drawBorder: false
                        }
                    },
                    x: {
                        ticks: { font: { size: 10 } },
                        grid: {
                            display: false
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }

    // Feature importance chart
    const featureCtx = document.getElementById('featureImportanceChart');
    if (featureCtx && !charts.featureImportance) {
        charts.featureImportance = new Chart(featureCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Minute', 'Road Type', 'Weather', 'Vehicle Type', 'Location', 'Speed', 'Age', 'Casualties'],
                datasets: [{
                    label: 'Importance',
                    data: [0.16, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08],
                    backgroundColor: '#800000'
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: { font: { size: 10 } },
                        grid: {
                            display: true,
                            drawBorder: false
                        }
                    },
                    y: {
                        ticks: { font: { size: 10 } },
                        grid: {
                            display: false
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }

    // Time series chart
    const timeSeriesCtx = document.getElementById('timeSeriesChart');
    if (timeSeriesCtx && !charts.timeSeries) {
        charts.timeSeries = new Chart(timeSeriesCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['0', '2', '4', '6', '8', '10', '12', '14', '16', '18', '20', '22'],
                datasets: [{
                    label: 'Accident Count',
                    data: [200, 150, 100, 250, 400, 380, 420, 400, 380, 450, 400, 300],
                    borderColor: '#800000',
                    backgroundColor: 'rgba(128, 0, 0, 0.1)',
                    fill: true,
                    tension: 0.4
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
                    y: {
                        beginAtZero: true,
                        ticks: { font: { size: 10 } },
                        grid: {
                            display: true,
                            drawBorder: false
                        }
                    },
                    x: {
                        ticks: { font: { size: 10 } },
                        grid: {
                            display: false
                        }
                    }
                },
                layout: {
                    padding: {
                        top: 10,
                        bottom: 10
                    }
                }
            }
        });
    }
}

// Apply filters
async function applyFilters() {
    const location = document.getElementById('filter-location').value;
    const weather = document.getElementById('filter-weather').value;
    const road = document.getElementById('filter-road').value;
    const vehicle = document.getElementById('filter-vehicle').value;
    
    const filterData = {
        location: location,
        weather: weather,
        road_type: road,
        vehicle_type: vehicle
    };
    
    try {
        const response = await fetch('/api/filter', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(filterData)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            updateDashboardWithFilteredData(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Filter error:', error);
        alert('Error applying filters. Please try again.');
    }
}

// Reset filters
function resetFilters() {
    document.getElementById('filter-location').value = 'all';
    document.getElementById('filter-weather').value = 'all';
    document.getElementById('filter-road').value = 'all';
    document.getElementById('filter-vehicle').value = 'all';
    
    // Reset to default data
    applyFilters();
}

// Update dashboard with filtered data
function updateDashboardWithFilteredData(data) {
    // Update stats cards
    document.querySelector('.stat-card.fatal .stat-value').textContent = data.severity_counts.fatal.toLocaleString();
    document.querySelector('.stat-card.serious .stat-value').textContent = data.severity_counts.serious.toLocaleString();
    document.querySelector('.stat-card.minor .stat-value').textContent = data.severity_counts.minor.toLocaleString();
    document.querySelector('.stat-card.today .stat-value').textContent = data.total_records.toLocaleString();
    
    // Update weather chart
    if (charts.weather) {
        charts.weather.data.datasets[0].data = data.weather_data.fatal;
        charts.weather.data.datasets[1].data = data.weather_data.serious;
        charts.weather.data.datasets[2].data = data.weather_data.minor;
        charts.weather.update();
    }
    
    // Update road chart
    if (charts.road) {
        charts.road.data.datasets[0].data = data.road_data.fatal;
        charts.road.data.datasets[1].data = data.road_data.serious;
        charts.road.data.datasets[2].data = data.road_data.minor;
        charts.road.update();
    }
}

// Load visualizations page
async function loadVisualizations() {
    // Fetch available plots
    try {
        const response = await fetch('/api/plots');
        const result = await response.json();
        
        if (response.ok && result.plots) {
            console.log('Available plots:', result.plots);
            // Images are already loaded via HTML img tags with the API endpoints
        }
    } catch (error) {
        console.error('Error loading visualizations:', error);
    }
}

// Show visualization phase
function showVizPhase(phase) {
    // Hide all phases
    document.querySelectorAll('.viz-phase').forEach(p => {
        p.classList.remove('active');
    });
    
    // Remove active class from all sidebar items
    document.querySelectorAll('.sidebar-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Show selected phase
    document.getElementById(`phase-${phase}`).classList.add('active');
    
    // Add active class to clicked sidebar item
    event.currentTarget.classList.add('active');
}
