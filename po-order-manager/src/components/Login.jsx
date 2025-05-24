import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box, TextField, Button, Typography, Tabs, Tab, Alert, CircularProgress
} from '@mui/material';
import axios from 'axios';

const Login = ({ setIsAuthenticated }) => {
  const [tabValue, setTabValue] = useState(0);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSignup = async () => {
    setError('');
    setLoading(true);
    try {
      await axios.post('http://localhost:8000/signup', { email, password });
      setTabValue(1);
      setError('Signup successful! Please sign in.');
    } catch (err) {
      setError(err.response?.data?.detail || 'Error during signup');
    }
    setLoading(false);
  };

  const handleSignin = async () => {
    setError('');
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/signin', { email, password });
      localStorage.setItem('token', response.data.access_token);
      setIsAuthenticated(true);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Error during signin');
    }
    setLoading(false);
  };

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '100vh',
        bgcolor: '#121212',
        p: 3,
      }}
    >
      <Typography variant="h4" sx={{ mb: 3, color: '#FFFFFF', fontWeight: 700 }}>
        🛒 PO Order Manager
      </Typography>
      <Box sx={{ width: '100%', maxWidth: 400, bgcolor: '#1E1E1E', p: 3, borderRadius: '12px' }}>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
          centered
          sx={{
            mb: 2,
            '& .MuiTab-root': {
              color: '#B0BEC5',
              '&.Mui-selected': { color: '#4CAF50' },
            },
            '& .MuiTabs-indicator': { backgroundColor: '#4CAF50' },
          }}
        >
          <Tab label="Sign Up" />
          <Tab label="Sign In" />
        </Tabs>
        {error && (
          <Alert severity={error.includes('successful') ? 'success' : 'error'} sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        <TextField
          label="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          fullWidth
          margin="normal"
          variant="outlined"
          type="email"
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': { borderColor: '#2E2E2E' },
              '&:hover fieldset': { borderColor: '#4CAF50' },
              '&.Mui-focused fieldset': { borderColor: '#4CAF50' },
            },
            '& .MuiInputLabel-root': { color: '#B0BEC5' },
            '& .MuiInputBase-input': { color: '#FFFFFF' },
          }}
        />
        <TextField
          label="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          fullWidth
          margin="normal"
          variant="outlined"
          type="password"
          sx={{
            '& .MuiOutlinedInput-root': {
              '& fieldset': { borderColor: '#2E2E2E' },
              '&:hover fieldset': { borderColor: '#4CAF50' },
              '&.Mui-focused fieldset': { borderColor: '#4CAF50' },
            },
            '& .MuiInputLabel-root': { color: '#B0BEC5' },
            '& .MuiInputBase-input': { color: '#FFFFFF' },
          }}
        />
        {tabValue === 0 ? (
          <Button
            variant="contained"
            fullWidth
            onClick={handleSignup}
            disabled={loading}
            sx={{ mt: 2, bgcolor: '#4CAF50', '&:hover': { bgcolor: '#45A049' } }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Sign Up'}
          </Button>
        ) : (
          <Button
            variant="contained"
            fullWidth
            onClick={handleSignin}
            disabled={loading}
            sx={{ mt: 2, bgcolor: '#4CAF50', '&:hover': { bgcolor: '#45A049' } }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Sign In'}
          </Button>
        )}
      </Box>
    </Box>
  );
};

export default Login;