import { useState } from 'react';
import {
  Box, Typography, TextField, Button, Select, MenuItem, FormControl, InputLabel,
  Grid, Divider, Card, CardContent
} from '@mui/material';
import { LocalizationProvider, DatePicker, TimePicker } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import axios from 'axios';
import { addDays } from 'date-fns';

const OrderForm = ({ setToast, setLoading }) => {
  const [formData, setFormData] = useState({
    product_name: '',
    category: '',
    price: '',
    quantity: '',
    customer_name: '',
    customer_phone: '',
    email: '',
    address: '',
    order_date: new Date(),
    order_time: new Date(),
    delivery_date: addDays(new Date(), 7),
    payment_method: 'COD',
    payment_status: 'Pending',
    order_status: 'Pending',
  });

  const handleChange = (name, value) => {
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = async (e) => {
  e.preventDefault();
  // ...validation

  setLoading(true);
  try {
    const order_date = new Date(
      formData.order_date.getFullYear(),
      formData.order_date.getMonth(),
      formData.order_date.getDate(),
      formData.order_time.getHours(),
      formData.order_time.getMinutes()
    );

    const orderData = {
      ...formData,
      price: parseFloat(formData.price),
      quantity: parseInt(formData.quantity),
      order_date: order_date.toISOString(), // ISO string
      delivery_date: formData.delivery_date.toISOString().slice(0, 10), // 'YYYY-MM-DD'
    };

    await axios.post('http://localhost:8000/orders/', orderData);
    setToast({ open: true, message: 'Order stored and notification queued', severity: 'success' });
    // ...reset form
  } catch (error) {
    setToast({ open: true, message: error?.response?.data?.detail || 'Error storing order', severity: 'error' });
  }
  setLoading(false);
};
  

  return (
    <LocalizationProvider dateAdapter={AdapterDateFns}>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        ✍️ Manual Order Entry
      </Typography>
      <Card sx={{ p: 2 }}>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Product Details
                </Typography>
                <TextField
                  label="Product Name *"
                  value={formData.product_name}
                  onChange={(e) => handleChange('product_name', e.target.value)}
                  fullWidth
                  margin="normal"
                  required
                  variant="outlined"
                />
                <TextField
                  label="Category"
                  value={formData.category}
                  onChange={(e) => handleChange('category', e.target.value)}
                  fullWidth
                  margin="normal"
                  variant="outlined"
                />
                <TextField
                  label="Price (₹) *"
                  type="number"
                  value={formData.price}
                  onChange={(e) => handleChange('price', e.target.value)}
                  fullWidth
                  margin="normal"
                  required
                  variant="outlined"
                  inputProps={{ min: 0.01, step: 0.01 }}
                />
                <TextField
                  label="Quantity *"
                  type="number"
                  value={formData.quantity}
                  onChange={(e) => handleChange('quantity', e.target.value)}
                  fullWidth
                  margin="normal"
                  required
                  variant="outlined"
                  inputProps={{ min: 1, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Customer Details
                </Typography>
                <TextField
                  label="Customer Name *"
                  value={formData.customer_name}
                  onChange={(e) => handleChange('customer_name', e.target.value)}
                  fullWidth
                  margin="normal"
                  required
                  variant="outlined"
                />
                <TextField
                  label="Customer Phone"
                  value={formData.customer_phone}
                  onChange={(e) => handleChange('customer_phone', e.target.value)}
                  fullWidth
                  margin="normal"
                  variant="outlined"
                />
                <TextField
                  label="Customer Email"
                  value={formData.email}
                  onChange={(e) => handleChange('email', e.target.value)}
                  fullWidth
                  margin="normal"
                  variant="outlined"
                />
                <TextField
                  label="Delivery Address *"
                  value={formData.address}
                  onChange={(e) => handleChange('address', e.target.value)}
                  fullWidth
                  margin="normal"
                  multiline
                  rows={3}
                  required
                  variant="outlined"
                />
              </Grid>
            </Grid>
            <Divider sx={{ my: 3, bgcolor: '#2E2E2E' }} />
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Dates
                </Typography>
                <DatePicker
                  label="Order Date *"
                  value={formData.order_date}
                  onChange={(value) => handleChange('order_date', value)}
                  renderInput={(params) => <TextField {...params} fullWidth margin="normal" required variant="outlined" />}
                />
                <TimePicker
                  label="Order Time *"
                  value={formData.order_time}
                  onChange={(value) => handleChange('order_time', value)}
                  renderInput={(params) => <TextField {...params} fullWidth margin="normal" required variant="outlined" />}
                />
                <DatePicker
                  label="Expected Delivery Date *"
                  value={formData.delivery_date}
                  onChange={(value) => handleChange('delivery_date', value)}
                  renderInput={(params) => <TextField {...params} fullWidth margin="normal" required variant="outlined" />}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Typography variant="subtitle1" sx={{ fontWeight: 500, mb: 1 }}>
                  Status
                </Typography>
                <FormControl fullWidth margin="normal" variant="outlined">
                  <InputLabel>Payment Method</InputLabel>
                  <Select
                    value={formData.payment_method}
                    onChange={(e) => handleChange('payment_method', e.target.value)}
                    label="Payment Method"
                  >
                    {['COD', 'Credit Card', 'UPI', 'Bank Transfer', 'Other'].map((option) => (
                      <MenuItem key={option} value={option}>{option}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth margin="normal" variant="outlined">
                  <InputLabel>Payment Status</InputLabel>
                  <Select
                    value={formData.payment_status}
                    onChange={(e) => handleChange('payment_status', e.target.value)}
                    label="Payment Status"
                  >
                    {['Paid', 'Unpaid', 'Pending'].map((option) => (
                      <MenuItem key={option} value={option}>{option}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
                <FormControl fullWidth margin="normal" variant="outlined">
                  <InputLabel>Order Status</InputLabel>
                  <Select
                    value={formData.order_status}
                    onChange={(e) => handleChange('order_status', e.target.value)}
                    label="Order Status"
                  >
                    {['Pending', 'Processing', 'Confirmed', 'Shipped', 'Delivered', 'Cancelled'].map((option) => (
                      <MenuItem key={option} value={option}>{option}</MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Grid>
            </Grid>
            <Button type="submit" variant="contained" fullWidth sx={{ mt: 3, py: 1.5 }}>
              Submit Order
            </Button>
          </form>
        </CardContent>
      </Card>
    </LocalizationProvider>
  );
};

export default OrderForm;