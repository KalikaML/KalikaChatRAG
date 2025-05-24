import { Box } from '@mui/material';

const TabPanel = ({ children, value, index, ...other }) => {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 2, bgcolor: 'background.default', borderRadius: '8px' }}>
          {children}
        </Box>
      )}
    </div>
  );
};

export default TabPanel;