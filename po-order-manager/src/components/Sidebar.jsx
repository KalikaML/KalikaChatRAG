import {
  Box, Typography, Switch, Slider, Button, Divider, Card, CardContent
} from '@mui/material';
import { format, addHours } from 'date-fns';
import { useNavigate } from 'react-router-dom';
const Sidebar = ({
  autoCheck, setAutoCheck, intervalMinutes, setIntervalMinutes,
  pendingCount, lastCheckTime, handleManualEmailCheck, handleSendPendingNotifications
}) => {
  const schedulerNextRun = new Date().setHours(new Date().getHours() + 2); // Scheduler runs every 2 hours
    const navigate = useNavigate();

  const handleLogout = () => {
    console.log('Logout button clicked'); // Debug: Confirm button click
    try {
      localStorage.removeItem('token');
      console.log('Token removed from localStorage'); // Debug: Confirm token removal
      setIsAuthenticated(false);
      console.log('isAuthenticated set to false'); // Debug: Confirm state update
      navigate('/login', { replace: true });
      console.log('Navigated to /login'); // Debug: Confirm navigation
    } catch (error) {
      console.error('Logout error:', error);
    }
  };
  return (
    <Box
      sx={{
        p: 3,
        bgcolor: 'background.paper',
        height: 'calc(100vh - 64px)',
        overflowY: 'auto',
      }}
    >
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        📊 Status
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">WhatsApp Sent</Typography>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>0</Typography>
        </CardContent>
      </Card>
      <Divider sx={{ my: 2, bgcolor: '#2E2E2E' }} />
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        ⚙️ Auto-Email Check
      </Typography>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography color="text.secondary">Enable Auto-Check</Typography>
        <Switch
          checked={autoCheck}
          onChange={(e) => setAutoCheck(e.target.checked)}
          sx={{
            '& .MuiSwitch-track': { backgroundColor: '#2E2E2E' },
            '& .MuiSwitch-thumb': { backgroundColor: autoCheck ? '#4CAF50' : '#B0BEC5' },
          }}
        />
      </Box>
      <Typography color="text.secondary" gutterBottom>Check Interval (minutes)</Typography>
      <Slider
        value={intervalMinutes}
        onChange={(e, value) => setIntervalMinutes(value)}
        min={5}
        max={120}
        step={5}
        valueLabelDisplay="auto"
        sx={{
          '& .MuiSlider-track': { backgroundColor: '#4CAF50' },
          '& .MuiSlider-rail': { backgroundColor: '#2E2E2E' },
          '& .MuiSlider-thumb': { backgroundColor: '#4CAF50' },
        }}
      />
      <Typography color="text.secondary" sx={{ mt: 1 }}>
        Last Check: {format(lastCheckTime, 'yyyy-MM-dd HH:mm:ss')}
      </Typography>
      {autoCheck && (
        <Typography color="text.secondary">
          Next Check: {format(new Date(lastCheckTime.getTime() + intervalMinutes * 60 * 1000), 'yyyy-MM-dd HH:mm:ss')}
        </Typography>
      )}
      <Divider sx={{ my: 2, bgcolor: '#2E2E2E' }} />
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        📧 Manual Email Check
      </Typography>
      <Button
        variant="contained"
        fullWidth
        onClick={handleManualEmailCheck}
        sx={{ mb: 2 }}
      >
        Check Emails Now
      </Button>
      <Divider sx={{ my: 2, bgcolor: '#2E2E2E' }} />
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        ⏰ Scheduler Status
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">Scheduler</Typography>
          <Typography variant="body1" sx={{ fontWeight: 500 }}>
            Running every 2 hours
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Next Run: {format(schedulerNextRun, 'yyyy-MM-dd HH:mm:ss')}
          </Typography>
        </CardContent>
      </Card>
      <Divider sx={{ my: 2, bgcolor: '#2E2E2E' }} />
      <Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
        📱 WhatsApp Notifications
      </Typography>
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary">Pending Orders</Typography>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            {pendingCount !== null ? pendingCount : 'Loading...'}
          </Typography>
        </CardContent>
      </Card>
      <Button
        variant="contained"
        fullWidth
        onClick={handleSendPendingNotifications}
        disabled={pendingCount === 0}
      >
        Send Pending Notifications
      </Button>
      <Divider sx={{ my: 2, bgcolor: '#2E2E2E' }} />
<Typography variant="h6" sx={{ fontWeight: 600, mb: 2 }}>
    🚪 Account
</Typography>
      <Button
        variant="outlined"
        fullWidth
        onClick={handleLogout}
        sx={{ borderColor: '#EF5350', color: '#EF5350', '&:hover': { borderColor: '#D32F2F', color: '#D32F2F' } }}
      >
        Logout
      </Button>
    </Box>
  );
};

export default Sidebar;