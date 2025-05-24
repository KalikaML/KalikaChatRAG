import { useState, useEffect } from 'react';
import {
  Box, Typography, Grid, Card, CardContent, Divider, CircularProgress
} from '@mui/material';
import { Bar, Pie } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import axios from 'axios';
import { DataGrid } from '@mui/x-data-grid';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  ArcElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = ({ setToast, setLoading }) => {
  const [metrics, setMetrics] = useState({ total_orders: 0, total_revenue_paid: 0, avg_order_value: 0 });
  const [statusData, setStatusData] = useState([]);
  const [categoryData, setCategoryData] = useState([]);
  const [recentOrders, setRecentOrders] = useState([]);
  const [loadingData, setLoadingData] = useState(false);

  useEffect(() => {
    const fetchDashboardData = async () => {
      setLoadingData(true);
      try {
        const metricsResponse = await axios.get('http://localhost:8000/dashboard/metrics');
        setMetrics(metricsResponse.data);

        const statusResponse = await axios.get('http://localhost:8000/dashboard/status-distribution');
        setStatusData(statusResponse.data);

        const categoryResponse = await axios.get('http://localhost:8000/dashboard/sales-by-category');
        setCategoryData(categoryResponse.data);

        const ordersResponse = await axios.get('http://localhost:8000/orders/', { params: { limit: 15 } });
        setRecentOrders(ordersResponse.data.map((order, index) => ({
          id: order.id,
          order_date: new Date(order.order_date).toLocaleString(),
          product_name: order.product_name,
          customer_name: order.customer_name,
          total_value: (order.price * order.quantity).toFixed(2),
          order_status: order.order_status,
          payment_status: order.payment_status,
          message_sent: order.message_sent ? 'Yes' : 'No',
        })));
      } catch (error) {
        setToast({ open: true, message: 'Error loading dashboard data', severity: 'error' });
      }
      setLoadingData(false);
    };
    fetchDashboardData();
  }, [setToast]);

  const statusChartData = {
    labels: statusData.map(item => item.status),
    datasets: [{
      label: 'Order Count',
      data: statusData.map(item => item.count),
      backgroundColor: '#4CAF50',
      borderColor: '#45A049',
      borderWidth: 1,
    }],
  };

  const categoryChartData = {
    labels: categoryData.map(item => item.category),
    datasets: [{
      data: categoryData.map(item => item.total_sales),
      backgroundColor: [
        '#4CAF50', '#F06292', '#42A5F5', '#FFCA28', '#66BB6A',
        '#EF5350', '#B0BEC5', '#FF7043', '#26A69A', '#AB47BC'
      ],
      borderColor: '#1E1E1E',
      borderWidth: 2,
    }],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#B0BEC5', font: { size: 14 } } },
      tooltip: { backgroundColor: '#2E2E2E', titleColor: '#FFFFFF', bodyColor: '#B0BEC5' },
    },
    scales: {
      x: { grid: { borderColor: '#2E2E2E', color: '#2E2E2E' }, ticks: { color: '#B0BEC5' } },
      y: { grid: { borderColor: '#2E2E2E', color: '#2E2E2E' }, ticks: { color: '#B0BEC5' } },
    },
  };

  const recentColumns = [
    { field: 'id', headerName: 'ID', width: 80 },
    { field: 'order_date', headerName: 'Order Date', width: 200 },
    { field: 'product_name', headerName: 'Product', width: 200 },
    { field: 'customer_name', headerName: 'Customer', width: 150 },
    { field: 'total_value', headerName: 'Total Value', width: 120 },
    { field: 'order_status', headerName: 'Status', width: 120 },
    { field: 'payment_status', headerName: 'Payment', width: 120 },
    { field: 'message_sent', headerName: 'Notified', width: 100 },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ fontWeight: 700 }}>
        📊 Dashboard
      </Typography>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        📈 Key Metrics
      </Typography>
      {loadingData ? (
        <CircularProgress sx={{ color: '#4CAF50' }} />
      ) : (
        <>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="body2" color="text.secondary">Total Orders</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>{metrics.total_orders}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="body2" color="text.secondary">Total Revenue</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>₹{metrics.total_revenue_paid.toFixed(2)}</Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Card>
                <CardContent>
                  <Typography variant="body2" color="text.secondary">Avg. Order Value</Typography>
                  <Typography variant="h5" sx={{ fontWeight: 700 }}>₹{metrics.avg_order_value.toFixed(2)}</Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
          <Divider sx={{ my: 3, bgcolor: '#2E2E2E' }} />
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            📊 Order Status
          </Typography>
          {statusData.length > 0 ? (
            <Card sx={{ p: 2, height: 300 }}>
              <Bar data={statusChartData} options={chartOptions} />
            </Card>
          ) : (
            <Typography color="text.secondary">No order status data.</Typography>
          )}
          <Divider sx={{ my: 3, bgcolor: '#2E2E2E' }} />
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            📦 Sales by Category
          </Typography>
          {categoryData.length > 0 ? (
            <Card sx={{ p: 2, height: 300, maxWidth: 400, mx: 'auto' }}>
              <Pie data={categoryChartData} options={chartOptions} />
            </Card>
          ) : (
            <Typography color="text.secondary">No category sales data.</Typography>
          )}
          <Divider sx={{ my: 3, bgcolor: '#2E2E2E' }} />
          <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
            📋 Recent Orders
          </Typography>
          {recentOrders.length > 0 ? (
            <Box sx={{ height: 400, width: '100%', mt: 2 }}>
              <DataGrid
                rows={recentOrders}
                columns={recentColumns}
                pageSize={15}
                rowsPerPageOptions={[15]}
                disableSelectionOnClick
                sx={{ borderRadius: '8px', overflow: 'hidden' }}
              />
            </Box>
          ) : (
            <Typography color="text.secondary">No orders to display.</Typography>
          )}
        </>
      )}
    </Box>
  );
};

export default Dashboard;