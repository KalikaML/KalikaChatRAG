import React from 'react';
import { Box, Typography, Button } from '@mui/material';

class ErrorBoundary extends React.Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Box sx={{ p: 3, textAlign: 'center', bgcolor: '#121212', minHeight: '100vh' }}>
          <Typography variant="h4" color="error" gutterBottom>
            Something went wrong
          </Typography>
          <Typography variant="body1" color="text.secondary" gutterBottom>
            An error occurred: {this.state.error?.message || 'Unknown error'}
          </Typography>
          <Button
            variant="contained"
            onClick={() => window.location.reload()}
            sx={{ mt: 2, bgcolor: '#4CAF50', '&:hover': { bgcolor: '#45A049' } }}
          >
            Reload Page
          </Button>
        </Box>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;