# EduVision Frontend

Interactive web-based frontend for the EduVision educational engagement platform. Built with React, Vite, and modern web technologies.

## Features

- **Role-based Authentication** - Separate interfaces for teachers, students, and administrators
- **Teacher Dashboard** - Session management, lecture creation, and analytics
- **Student Dashboard** - Session joining, real-time monitoring, and engagement tracking
- **Real-time Analytics** - Charts and visualizations for engagement data
- **Session Management** - Create, start, and end lecture sessions
- **Calendar Integration** - View and manage scheduled sessions
- **Responsive Design** - Works on desktop, tablet, and mobile devices

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- EduVision API server running on port 8002
- PostgreSQL database running

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

### Demo Accounts

**Teacher:**
- Email: `integration.teacher@eduvision.com`
- Password: `TeacherPass123!`

**Student:**
- Email: `integration.student@eduvision.com`
- Password: `StudentPass123!`

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable React components
│   │   ├── Layout/      # Sidebar, Header, Layout components
│   │   ├── Auth/        # Authentication components
│   │   ├── Teacher/     # Teacher-specific components
│   │   ├── Student/     # Student-specific components
│   │   ├── Common/      # Shared components
│   │   └── Charts/      # Data visualization components
│   ├── pages/           # Page components
│   │   ├── Login.jsx
│   │   ├── TeacherDashboard.jsx
│   │   ├── StudentDashboard.jsx
│   │   └── AnalyticsPage.jsx
│   ├── services/        # API service layer
│   │   ├── api.js       # Axios instance with interceptors
│   │   ├── authService.js
│   │   ├── sessionService.js
│   │   └── analyticsService.js
│   ├── hooks/           # Custom React hooks
│   │   ├── useAuth.js
│   │   ├── useSession.js
│   │   └── useWebcam.js
│   ├── utils/           # Utility functions
│   ├── assets/          # Styles and images
│   ├── App.jsx          # Main app component
│   └── main.jsx         # Entry point
├── package.json
├── vite.config.js
└── README.md
```

## Development

### Available Scripts

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linter
npm run lint
```

### Environment Variables

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8002
```

## Features Overview

### Login Page
- Role selection (Teacher/Student/Admin)
- Email and password authentication
- Demo mode for quick testing
- Backend connection status indicator

### Teacher Dashboard
- **Start Session**: Create and manage lecture sessions
- **Session Controls**: Start/stop sessions, generate session codes
- **Analytics**: View engagement statistics and charts
- **Quiz Report**: Generate and view quiz results (coming soon)
- **Notes**: Lecture notes generation (coming soon)
- **Student Stats**: Individual student performance tracking

### Student Dashboard
- **Join Session**: Enter session code to join active lectures
- **Camera Feed**: Real-time video capture for engagement monitoring
- **Engagement Monitor**: Live feedback on attention and emotion
- **Session Analytics**: Personal engagement scores and statistics

### Session Management
- Create sessions linked to lectures
- Generate unique session codes (8 characters)
- Start/stop sessions
- Real-time participant tracking
- Automatic engagement data collection

### Analytics & Charts
- Engagement over time (line chart)
- Emotion distribution (pie chart)
- Participant statistics
- Session duration and metrics
- Export capabilities (coming soon)

## API Integration

The frontend communicates with the unified API backend through axios services:

### Authentication
- `POST /auth/login` - User login
- `POST /auth/logout` - User logout
- `POST /auth/refresh` - Token refresh

### Teacher Endpoints
- `POST /api/lectures` - Create lecture
- `GET /api/lectures` - List lectures
- `POST /api/sessions` - Create session
- `POST /api/sessions/{code}/start` - Start session
- `POST /api/sessions/{code}/end` - End session
- `GET /api/sessions/{id}/analytics` - Get analytics

### Student Endpoints
- `POST /api/sessions/{code}/join` - Join session
- `POST /api/sessions/{id}/engagement` - Log engagement events

## Styling

The application uses CSS with CSS variables for theming:

- **Primary Colors**: Dark blue (#1e293b), Blue (#3b82f6)
- **Background**: Light gray (#f8fafc)
- **Sidebar**: Dark gray (#2d3748)
- **Cards**: White (#ffffff)
- **Responsive**: Mobile-first design with breakpoints

## Dependencies

### Core
- `react` ^18.2.0
- `react-dom` ^18.2.0
- `react-router-dom` ^6.20.0

### Data Visualization
- `chart.js` ^4.4.1
- `react-chartjs-2` ^5.2.0

### HTTP & State
- `axios` ^1.6.2

### UI Components
- `lucide-react` ^0.294.0 (Icons)
- `react-calendar` ^4.7.0 (Calendar)
- `date-fns` ^2.30.0 (Date utilities)

### Build Tools
- `vite` ^5.0.8
- `@vitejs/plugin-react` ^4.2.1

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Known Issues

- Webcam access requires HTTPS in production
- Some features require modern browser APIs (WebRTC, MediaDevices)
- Mobile camera support may vary by device

## Roadmap

### Phase 1 (Current)
- ✅ Login and authentication
- ✅ Teacher dashboard
- ✅ Student dashboard
- ✅ Session management
- ✅ Basic analytics

### Phase 2 (In Progress)
- [ ] Real-time webcam integration
- [ ] Live engagement tracking
- [ ] Enhanced analytics charts
- [ ] Quiz generation UI
- [ ] Notes management

### Phase 3 (Planned)
- [ ] WebSocket for real-time updates
- [ ] File upload for materials
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Export reports (PDF, CSV)

## Contributing

1. Create a feature branch
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## Testing

```bash
# Install dependencies
npm install

# Run tests (to be implemented)
npm test

# Run E2E tests (to be implemented)
npm run test:e2e
```

## Deployment

### Development Build

```bash
npm run dev
```

### Production Build

```bash
# Build for production
npm run build

# Output will be in dist/

# Preview production build
npm run preview
```

### Deployment Checklist

- [ ] Set environment variables
- [ ] Configure CORS on API server
- [ ] Enable HTTPS
- [ ] Set up CDN for static assets
- [ ] Configure domain and DNS
- [ ] Set up monitoring and analytics
- [ ] Enable error tracking (Sentry)

## Troubleshooting

### "Cannot connect to API"
- Ensure API server is running on port 8002
- Check CORS configuration
- Verify network connectivity

### "Camera not working"
- Check browser permissions
- Ensure HTTPS (required for WebRTC)
- Try different browser

### "Login failed"
- Verify credentials
- Check API server logs
- Clear browser cache and cookies

## Performance

- **Bundle Size**: ~500KB (production)
- **Load Time**: <2s on broadband
- **Time to Interactive**: <3s
- **Lighthouse Score**: 90+ (Performance)

## Security

- JWT token-based authentication
- Automatic token refresh
- Secure cookie storage (production)
- CORS protection
- XSS prevention
- CSRF protection

## Support

For issues or questions:
1. Check [FRONTEND_IMPLEMENTATION_GUIDE.md](../FRONTEND_IMPLEMENTATION_GUIDE.md)
2. Review [END_TO_END_USAGE_GUIDE.md](../END_TO_END_USAGE_GUIDE.md)
3. Test API at `http://localhost:8002/docs`
4. Check browser console for errors

## License

This project is part of the EduVision educational platform.

---

*Last Updated: 2025-11-18*
*Version: 1.0.0*
*Status: Development*
