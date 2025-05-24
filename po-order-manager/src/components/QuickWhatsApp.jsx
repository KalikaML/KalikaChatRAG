import { useState } from 'react';
import { Box, Typography, TextField, Button, Checkbox, FormControlLabel, Grid, Card, CardContent } from '@mui/material';
import axios from 'axios';

const QuickWhatsApp = ({ setToast, setLoading }) => {
  const contactDict = {
    Narayan: '+919067847003',
    Rani_Bhise: '+917070242402',
    Abhishek: '+919284625240',
    Damini: '+917499353409',
    Sandeep: '+919850030215',
    Chandrakant: '+919665934999',
    Vikas_Kumbharkar: '+919284238738',
    bhavin:'+916353761393'
  };

  const [message, setMessage] = useState('Hello! Quick update: ');
  const [selectedContacts, setSelectedContacts] = useState([]);

  const handleSubmit = async () => {
    if (!message.trim()) {
      setToast({ open: true, message: 'Please enter a message', severity: 'warning' });
      return;
    }
    if (selectedContacts.length === 0) {
      setToast({ open: true, message: 'No contacts selected', severity: 'warning' });
      return;
    }

    setLoading(true);
    try {
      await axios.post('http://localhost:8000/whatsapp/quick-send', {
        contact_names: selectedContacts,
        message,
      });
      setToast({ open: true, message: 'WhatsApp messages queued', severity: 'success' });
      setMessage('Hello! Quick update: ');
      setSelectedContacts([]);
    } catch (error) {
      setToast({ open: true, message: 'Error sending WhatsApp messages', severity: 'error' });
    }
    setLoading(false);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
        📱 Quick WhatsApp
      </Typography>
      <Card sx={{ p: 2 }}>
        <CardContent>
          <TextField
            label="Custom Message"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            fullWidth
            multiline
            rows={4}
            margin="normal"
            variant="outlined"
          />
          <Typography variant="subtitle1" sx={{ fontWeight: 500, mt: 2 }}>
            Select Contacts
          </Typography>
          <Grid container spacing={2}>
            {Object.entries(contactDict).map(([name, phone]) => (
              <Grid item xs={12} sm={6} md={4} key={name}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={selectedContacts.includes(name)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedContacts([...selectedContacts, name]);
                        } else {
                          setSelectedContacts(selectedContacts.filter((c) => c !== name));
                        }
                      }}
                      sx={{
                        color: '#B0BEC5',
                        '&.Mui-checked': { color: '#4CAF50' },
                      }}
                    />
                  }
                  label={`${name} (${phone})`}
                  sx={{ color: 'text.secondary' }}
                />
              </Grid>
            ))}
          </Grid>
          <Button
            variant="contained"
            fullWidth
            sx={{ mt: 3, py: 1.5 }}
            onClick={handleSubmit}
          >
            Send WhatsApp
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default QuickWhatsApp;
