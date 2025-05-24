import { useState, useEffect } from 'react';
import { Routes, Route, Navigate, BrowserRouter } from 'react-router-dom';
import {
  AppBar, Toolbar, Typography, Box, Tabs, Tab, Drawer, CircularProgress, Snackbar, Alert,
  CssBaseline, Container, TextField
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Sidebar from './components/Sidebar';
import OrderForm from './components/OrderForm';
import QuickWhatsApp from './components/QuickWhatsApp';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import TabPanel from './components/TabPanel';
import ErrorBoundary from './components/ErrorBoundary';
import { DataGrid } from '@mui/x-data-grid';
import axios from 'axios';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#4CAF50' },
    secondary: { main: '#F06292' },
    background: {
      default: '#121212',
      paper: '#1E1E1E',
    },
    text: {
      primary: '#FFFFFF',
      secondary: '#B0BEC5',
    },
    error: { main: '#EF5350' },
    success: { main: '#66BB6A' },
    info: { main: '#42A5F5' },
    warning: { main: '#FFCA28' },
  },
  typography: {
    fontFamily: "'Inter', 'Roboto', 'Arial', sans-serif",
    h4: { fontWeight: 700 },
    h6: { fontWeight: 600 },
    body1: { fontSize: '1rem' },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          background: 'linear-gradient(90deg, #1E1E1E 0%, #2E2E2E 100%)',
          boxShadow: '0 2px 4px rgba(0,0,0,0.5)',
        },
      },
    },
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: '#1E1E1E',
          borderRight: '1px solid #2E2E2E',
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          backgroundColor: '#1E1E1E',
          borderRadius: '8px',
          padding: '4px',
        },
        indicator: {
          backgroundColor: '#4CAF50',
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
          color: '#B0BEC5',
          '&.Mui-selected': {
            color: '#FFFFFF',
          },
          '&:hover': {
            color: '#4CAF50',
            backgroundColor: '#2E2E2E',
            borderRadius: '4px',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: '8px',
          padding: '8px 16px',
          transition: 'all 0.3s ease',
          '&:hover': {
            transform: 'translateY(-1px)',
            boxShadow: '0 4px 8px rgba(0,0,0,0.3)',
          },
        },
        contained: {
          backgroundColor: '#4CAF50',
          color: '#FFFFFF',
          '&:hover': {
            backgroundColor: '#45A049',
          },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1E1E1E',
          borderRadius: '12px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
          transition: 'transform 0.3s ease',
          '&:hover': {
            transform: 'translateY(-4px)',
          },
        },
      },
    },
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#2E2E2E',
            },
            '&:hover fieldset': {
              borderColor: '#4CAF50',
            },
            '&.Mui-focused fieldset': {
              borderColor: '#4CAF50',
            },
          },
        },
      },
    },
    MuiDataGrid: {
      styleOverrides: {
        root: {
          backgroundColor: '#1E1E1E',
          border: 'none',
          '& .MuiDataGrid-columnHeaders': {
            backgroundColor: '#2E2E2E',
            color: '#FFFFFF',
          },
          '& .MuiDataGrid-cell': {
            borderBottom: '1px solid #2E2E2E',
            color: '#B0BEC5',
          },
          '& .MuiDataGrid-row:hover': {
            backgroundColor: '#2E2E2E',
          },
        },
      },
    },
  },
});

const ProtectedRoute = ({ children, isAuthenticated }) => {
  return isAuthenticated ? children : <Navigate to="/login" />;
};

const App = () => {
  const [tabValue, setTabValue] = useState(0);
  const [autoCheck, setAutoCheck] = useState(false);
  const [intervalMinutes, setIntervalMinutes] = useState(30);
  const [pendingCount, setPendingCount] = useState(null);
  const [lastCheckTime, setLastCheckTime] = useState(new Date());
  const [emailQuery, setEmailQuery] = useState('PO released // Consumable items,PO copy,import po,RFQ-Polybag,PFA PO,Purchase Order FOR,Purchase Order_');
  const [recentOrders, setRecentOrders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState({ open: false, message: '', severity: 'info' });
  const [isAuthenticated, setIsAuthenticated] = useState(!!localStorage.getItem('token'));

  // Set axios default headers for authenticated requests
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    } else {
      delete axios.defaults.headers.common['Authorization'];
    }
  }, [isAuthenticated]);

  // Fetch recent orders and pending count
  useEffect(() => {
    if (!isAuthenticated) return;
    const fetchData = async () => {
      setLoading(true);
      try {
        const ordersResponse = await axios.get('http://localhost:8000/orders/', { params: { limit: 10 } });
        setRecentOrders(ordersResponse.data.map((order) => ({
          id: order.id,
          order_date: new Date(order.order_date).toLocaleString(),
          product_name: order.product_name,
          customer_name: order.customer_name,
          price: order.price,
          quantity: order.quantity,
          order_status: order.order_status,
          message_sent: order.message_sent ? 'Yes' : 'No',
        })));

        const pendingResponse = await axios.get('http://localhost:8000/orders/', {
          params: { message_sent: false },
        });
        setPendingCount(pendingResponse.data.length);
      } catch (error) {
        setToast({ open: true, message: 'Error fetching data', severity: 'error' });
      }
      setLoading(false);
    };
    fetchData();
  }, [isAuthenticated]);

  // Handle auto-check scheduling
  useEffect(() => {
    if (autoCheck && isAuthenticated) {
      const interval = setInterval(async () => {
        try {
          await axios.post('http://localhost:8000/emails/process-orders', {
            subject_query: emailQuery,
            only_recent_days: 7,
            mark_as_read_after_extraction: true,
          });
          setLastCheckTime(new Date());
          setToast({ open: true, message: 'Auto-email check triggered', severity: 'success' });
        } catch (error) {
          setToast({ open: true, message: 'Auto-check failed', severity: 'error' });
        }
      }, intervalMinutes * 60 * 1000);
      return () => clearInterval(interval);
    }
  }, [autoCheck, intervalMinutes, emailQuery, isAuthenticated]);

  const handleManualEmailCheck = async () => {
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/emails/process-orders', {
        subject_query: emailQuery,
        only_recent_days: 7,
        mark_as_read_after_extraction: true,
      });
      await axios.post('http://localhost:8000/whatsapp/send-pending-db');
      setToast({ open: true, message: 'Email check and notifications queued', severity: 'success' });
      const ordersResponse = await axios.get('http://localhost:8000/orders/', { params: { limit: 10 } });
      setRecentOrders(ordersResponse.data.map((order) => ({
        id: order.id,
        order_date: new Date(order.order_date).toLocaleString(),
        product_name: order.product_name,
        customer_name: order.customer_name,
        price: order.price,
        quantity: order.quantity,
        order_status: order.order_status,
        message_sent: order.message_sent ? 'Yes' : 'No',
      })));
    } catch (error) {
      setToast({ open: true, message: 'Error processing emails', severity: 'error' });
    }
    setLoading(false);
  };

  const handleSendPendingNotifications = async () => {
    if (pendingCount === 0) {
      setToast({ open: true, message: 'No pending notifications', severity: 'info' });
      return;
    }
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/whatsapp/send-pending-db');
      setToast({ open: true, message: 'Pending notifications queued', severity: 'success' });
      setPendingCount(0);
    } catch (error) {
      setToast({ open: true, message: 'Error sending notifications', severity: 'error' });
    }
    setLoading(false);
  };

  const columns = [
    { field: 'order_date', headerName: 'Order Date', width: 200 },
    { field: 'product_name', headerName: 'Product', width: 200 },
    { field: 'customer_name', headerName: 'Customer', width: 150 },
    { field: 'price', headerName: 'Price', width: 100 },
    { field: 'quantity', headerName: 'Qty', width: 80 },
    { field: 'order_status', headerName: 'Status', width: 120 },
    { field: 'message_sent', headerName: 'Notified', width: 100 },
  ];

  const MainApp = () => (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" noWrap sx={{ fontWeight: 700 }}>
            🛒 PO Order Manager
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: 300,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: 300, boxSizing: 'border-box', mt: 8 },
        }}
      >
        <Sidebar
          autoCheck={autoCheck}
          setAutoCheck={setAutoCheck}
          intervalMinutes={intervalMinutes}
          setIntervalMinutes={setIntervalMinutes}
          pendingCount={pendingCount}
          lastCheckTime={lastCheckTime}
          handleManualEmailCheck={handleManualEmailCheck}
          handleSendPendingNotifications={handleSendPendingNotifications}
          setIsAuthenticated={setIsAuthenticated}
        />
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3, mt: 8, bgcolor: 'background.default' }}>
        <Container maxWidth="xl">
          <Tabs
            value={tabValue}
            onChange={(e, newValue) => setTabValue(newValue)}
            aria-label="PO Management Tabs"
            sx={{ mb: 3 }}
          >
            <Tab label="📬 Email PO Processing" />
            <Tab label="📝 Manual PO Entry" />
            <Tab label="📞 Quick WhatsApp" />
            <Tab label="📊 Dashboard" />
          </Tabs>
          <TabPanel value={tabValue} index={0}>
            <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
              📧 Email Order Search
            </Typography>
            <TextField
              label="Email Subjects (comma-separated)"
              value={emailQuery}
              onChange={(e) => setEmailQuery(e.target.value)}
              fullWidth
              margin="normal"
              variant="outlined"
              sx={{ maxWidth: 600 }}
              helperText="Enter subjects to search, separated by commas (e.g., Purchase Order,PO copy)"
            />
            <Typography variant="body1" gutterBottom sx={{ color: 'text.secondary' }}>
              Current search subjects: <strong>{emailQuery}</strong>
            </Typography>
            <Alert severity="info" sx={{ mt: 2, borderRadius: '8px' }}>
              Use 'Manually Check Emails' in sidebar for immediate processing. Scheduler runs every 2 hours for these subjects.
            </Alert>
            <Typography variant="h6" gutterBottom sx={{ mt: 4, fontWeight: 600 }}>
              📋 Recent Orders
            </Typography>
            {loading ? (
              <CircularProgress sx={{ color: '#4CAF50' }} />
            ) : recentOrders && recentOrders.length > 0 ? (
              <Box sx={{ height: 400, width: '100%', mt: 2 }}>
                <DataGrid
                  rows={recentOrders}
                  columns={columns}
                  pageSizeOptions={[10]}
                  disableSelectionOnClick
                  sx={{ borderRadius: '8px', overflow: 'hidden' }}
                />
              </Box>
            ) : (
              <Typography color="text.secondary">No orders found.</Typography>
            )}
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <OrderForm setToast={setToast} setLoading={setLoading} />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <QuickWhatsApp setToast={setToast} setLoading={setLoading} />
          </TabPanel>
          <TabPanel value={tabValue} index={3}>
            <Dashboard setToast={setToast} setLoading={setLoading} />
          </TabPanel>
        </Container>
      </Box>
      <Snackbar
        open={toast.open}
        autoHideDuration={3000}
        onClose={() => setToast({ ...toast, open: false })}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert
          onClose={() => setToast({ ...toast, open: false })}
          severity={toast.severity}
          sx={{ width: '100%', borderRadius: '8px' }}
        >
          {toast.message}
        </Alert>
      </Snackbar>
    </Box>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <ErrorBoundary>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login setIsAuthenticated={setIsAuthenticated} />} />
            <Route
              path="/dashboard"
              element={
                <ProtectedRoute isAuthenticated={isAuthenticated}>
                  <MainApp />
                </ProtectedRoute>
              }
            />
            <Route path="/" element={<Navigate to={isAuthenticated ? "/dashboard" : "/login"} />} />
          </Routes>
        </BrowserRouter>
      </ErrorBoundary>
    </ThemeProvider>
  );
};

export default App;