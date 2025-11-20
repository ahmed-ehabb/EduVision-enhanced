import { useState, useEffect, useRef } from 'react'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import { sessionService } from '../services/sessionService'
import './StudentDashboard.css'

export default function StudentDashboard() {
  const [sessionCode, setSessionCode] = useState('')
  const [currentSession, setCurrentSession] = useState(null)
  const [isMonitoring, setIsMonitoring] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  const [stats, setStats] = useState({
    sessionsJoined: 0,
    avgEngagement: 0,
    totalDuration: 0,
  })

  const videoRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => {
    fetchStudentStats()

    return () => {
      stopCamera()
    }
  }, [])

  const fetchStudentStats = async () => {
    try {
      // Fetch student's session history
      const response = await sessionService.getAnalytics()
      // Calculate stats from response
      setStats({
        sessionsJoined: response.total_sessions || 0,
        avgEngagement: Math.round(response.avg_engagement || 0),
        totalDuration: Math.round(response.total_duration_minutes || 0),
      })
    } catch (error) {
      console.error('Failed to fetch student stats:', error)
    }
  }

  const handleJoinSession = async (e) => {
    e.preventDefault()

    if (!sessionCode.trim()) {
      alert('Please enter a session code')
      return
    }

    try {
      const session = await sessionService.joinSession(sessionCode.toUpperCase())
      setCurrentSession(session)
      setSessionCode('')
    } catch (error) {
      if (error.response?.status === 404) {
        alert('Session not found. Please check the code and try again.')
      } else if (error.response?.status === 400) {
        alert('Invalid session code or session is not active.')
      } else {
        alert('Failed to join session. Please try again.')
      }
      console.error('Failed to join session:', error)
    }
  }

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      })

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setIsMonitoring(true)
        setCameraError(null)

        // Here you would start sending frames to the backend for analysis
        // This would integrate with the student-module's engagement tracking
      }
    } catch (error) {
      console.error('Camera access error:', error)
      setCameraError('Unable to access camera. Please check permissions.')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsMonitoring(false)
  }

  const handleLeaveSession = () => {
    stopCamera()
    setCurrentSession(null)
  }

  return (
    <div className="dashboard-layout">
      <Sidebar role="student" />
      <div className="dashboard-main">
        <Header
          title="Student Dashboard"
          subtitle="Join a session to start tracking your engagement"
        />

        <div className="dashboard-content">
          {!currentSession ? (
            <>
              {/* Stats Grid */}
              <div className="grid grid-3">
                <div className="stat-card">
                  <div className="stat-label">Sessions Joined</div>
                  <div className="stat-value">{stats.sessionsJoined}</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Avg Engagement</div>
                  <div className="stat-value">{stats.avgEngagement}%</div>
                </div>
                <div className="stat-card">
                  <div className="stat-label">Total Duration</div>
                  <div className="stat-value">{stats.totalDuration} min</div>
                </div>
              </div>

              {/* Join Session Card */}
              <div className="card join-session-card">
                <div className="card-header">
                  <h3 className="card-title">Join a Session</h3>
                  <p className="card-subtitle">Enter the session code provided by your teacher</p>
                </div>
                <form onSubmit={handleJoinSession} className="card-body">
                  <div className="join-session-form">
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <input
                        type="text"
                        className="form-input session-code-input"
                        value={sessionCode}
                        onChange={(e) => setSessionCode(e.target.value.toUpperCase())}
                        placeholder="Enter 8-character code"
                        maxLength={8}
                        style={{ textAlign: 'center', fontSize: '1.5rem', letterSpacing: '0.2em' }}
                      />
                    </div>
                    <button type="submit" className="btn btn-primary btn-lg">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4" />
                        <polyline points="10 17 15 12 10 7" />
                        <line x1="15" y1="12" x2="3" y2="12" />
                      </svg>
                      Join Session
                    </button>
                  </div>
                </form>
              </div>

              {/* Instructions */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">How It Works</h3>
                </div>
                <div className="card-body">
                  <div className="instructions-list">
                    <div className="instruction-item">
                      <div className="instruction-number">1</div>
                      <div className="instruction-content">
                        <h4>Get Session Code</h4>
                        <p>Your teacher will display an 8-character session code</p>
                      </div>
                    </div>
                    <div className="instruction-item">
                      <div className="instruction-number">2</div>
                      <div className="instruction-content">
                        <h4>Join Session</h4>
                        <p>Enter the code above and click "Join Session"</p>
                      </div>
                    </div>
                    <div className="instruction-item">
                      <div className="instruction-number">3</div>
                      <div className="instruction-content">
                        <h4>Allow Camera Access</h4>
                        <p>Grant camera permission when prompted by your browser</p>
                      </div>
                    </div>
                    <div className="instruction-item">
                      <div className="instruction-number">4</div>
                      <div className="instruction-content">
                        <h4>Start Monitoring</h4>
                        <p>Click "Start Analysis" to begin engagement tracking</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Active Session View */}
              <div className="active-session-header">
                <div className="active-session-info">
                  <h2>Active Session</h2>
                  <div className="session-code-display">
                    <span className="label">Code:</span>
                    <span className="code">{currentSession.session_code}</span>
                  </div>
                </div>
                <button className="btn btn-danger" onClick={handleLeaveSession}>
                  Leave Session
                </button>
              </div>

              <div className="grid grid-2">
                {/* Camera Feed */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Camera Feed</h3>
                    <p className="card-subtitle">
                      {isMonitoring ? 'Monitoring active' : 'Ready to start'}
                    </p>
                  </div>
                  <div className="card-body">
                    <div className="camera-container">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="camera-video"
                      />
                      {!isMonitoring && (
                        <div className="camera-placeholder">
                          <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                            <circle cx="12" cy="13" r="4" />
                          </svg>
                          <p>Click "Start Analysis" to begin</p>
                        </div>
                      )}
                    </div>
                    {cameraError && (
                      <div className="alert alert-danger" style={{ marginTop: '1rem' }}>
                        {cameraError}
                      </div>
                    )}
                    <div className="camera-controls">
                      {!isMonitoring ? (
                        <button className="btn btn-primary btn-lg w-full" onClick={startCamera}>
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polygon points="5 3 19 12 5 21 5 3" />
                          </svg>
                          Start Analysis
                        </button>
                      ) : (
                        <button className="btn btn-danger btn-lg w-full" onClick={stopCamera}>
                          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <rect x="6" y="6" width="12" height="12" />
                          </svg>
                          Stop Analysis
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Engagement Stats */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Your Engagement</h3>
                    <p className="card-subtitle">Real-time monitoring data</p>
                  </div>
                  <div className="card-body">
                    <div className="engagement-stats">
                      <div className="engagement-stat">
                        <div className="engagement-stat-label">Current Status</div>
                        <div className="engagement-stat-value">
                          {isMonitoring ? (
                            <span className="status status-active">
                              <span className="status-dot"></span>
                              Monitoring
                            </span>
                          ) : (
                            <span className="status status-inactive">
                              <span className="status-dot"></span>
                              Not Started
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="engagement-stat">
                        <div className="engagement-stat-label">Engagement Score</div>
                        <div className="engagement-stat-value">--</div>
                      </div>
                      <div className="engagement-stat">
                        <div className="engagement-stat-label">Detected Emotion</div>
                        <div className="engagement-stat-value">--</div>
                      </div>
                      <div className="engagement-stat">
                        <div className="engagement-stat-label">Concentration</div>
                        <div className="engagement-stat-value">--</div>
                      </div>
                      <div className="engagement-stat">
                        <div className="engagement-stat-label">Attention Status</div>
                        <div className="engagement-stat-value">--</div>
                      </div>
                    </div>
                    {!isMonitoring && (
                      <div className="alert alert-info" style={{ marginTop: '1rem' }}>
                        Start analysis to see real-time engagement metrics
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
