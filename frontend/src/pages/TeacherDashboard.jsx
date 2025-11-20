import { useState, useEffect } from 'react'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import api from '../services/api'
import './TeacherDashboard.css'

export default function TeacherDashboard() {
  const [stats, setStats] = useState({
    totalLectures: 0,
    activeSessions: 0,
    totalStudents: 0,
    avgEngagement: 0,
  })
  const [recentSessions, setRecentSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newLecture, setNewLecture] = useState({
    title: '',
    description: '',
    course_code: '',
    course_name: '',
  })

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      setLoading(true)
      // Fetch lectures
      const lecturesRes = await api.get('/api/lectures')
      const lectures = lecturesRes.data

      // Fetch sessions
      const sessionsRes = await api.get('/api/sessions')
      const sessions = sessionsRes.data

      // Calculate stats
      const activeSessions = sessions.filter(s => s.status === 'active').length
      const completedSessions = sessions.filter(s => s.status === 'completed')

      // Get recent sessions (last 5)
      const recent = sessions
        .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
        .slice(0, 5)

      setStats({
        totalLectures: lectures.length,
        activeSessions: activeSessions,
        totalStudents: calculateTotalStudents(sessions),
        avgEngagement: calculateAvgEngagement(completedSessions),
      })

      setRecentSessions(recent)
    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const calculateTotalStudents = (sessions) => {
    const uniqueStudents = new Set()
    sessions.forEach(session => {
      if (session.participants) {
        session.participants.forEach(p => uniqueStudents.add(p.student_id))
      }
    })
    return uniqueStudents.size
  }

  const calculateAvgEngagement = (sessions) => {
    if (sessions.length === 0) return 0
    const total = sessions.reduce((sum, s) => sum + (s.avg_engagement || 0), 0)
    return Math.round(total / sessions.length)
  }

  const handleCreateLecture = async (e) => {
    e.preventDefault()
    try {
      await api.post('/api/lectures', newLecture)
      setShowCreateModal(false)
      setNewLecture({ title: '', description: '', course_code: '', course_name: '' })
      fetchDashboardData()
    } catch (error) {
      console.error('Failed to create lecture:', error)
      alert('Failed to create lecture. Please try again.')
    }
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    const date = new Date(dateString)
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getStatusClass = (status) => {
    switch (status) {
      case 'active':
        return 'status-active'
      case 'completed':
        return 'status-completed'
      case 'scheduled':
        return 'status-pending'
      default:
        return 'status-inactive'
    }
  }

  if (loading) {
    return (
      <div className="dashboard-layout">
        <Sidebar role="teacher" />
        <div className="dashboard-main">
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <div className="spinner"></div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard-layout">
      <Sidebar role="teacher" />
      <div className="dashboard-main">
        <Header
          title="Dashboard"
          subtitle="Welcome back! Here's what's happening with your lectures today."
          actions={
            <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              Create Lecture
            </button>
          }
        />

        <div className="dashboard-content">
          {/* Stats Grid */}
          <div className="grid grid-4">
            <div className="stat-card">
              <div className="stat-label">Total Lectures</div>
              <div className="stat-value">{stats.totalLectures}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Active Sessions</div>
              <div className="stat-value">{stats.activeSessions}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Total Students</div>
              <div className="stat-value">{stats.totalStudents}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Avg Engagement</div>
              <div className="stat-value">{stats.avgEngagement}%</div>
            </div>
          </div>

          {/* Recent Sessions */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Recent Sessions</h3>
              <p className="card-subtitle">Your latest lecture sessions</p>
            </div>
            <div className="card-body">
              {recentSessions.length === 0 ? (
                <div className="empty-state">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                  <h4>No sessions yet</h4>
                  <p>Create a lecture and start your first session</p>
                  <button className="btn btn-primary btn-sm" onClick={() => setShowCreateModal(true)}>
                    Create Lecture
                  </button>
                </div>
              ) : (
                <div className="table-container">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Session Code</th>
                        <th>Status</th>
                        <th>Participants</th>
                        <th>Started At</th>
                        <th>Duration</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentSessions.map((session) => (
                        <tr key={session.session_id}>
                          <td>
                            <span className="session-code">{session.session_code}</span>
                          </td>
                          <td>
                            <span className={`status ${getStatusClass(session.status)}`}>
                              <span className="status-dot"></span>
                              {session.status}
                            </span>
                          </td>
                          <td>{session.participant_count || 0}</td>
                          <td>{formatDate(session.started_at)}</td>
                          <td>
                            {session.duration_minutes
                              ? `${session.duration_minutes} min`
                              : session.status === 'active'
                              ? 'In progress'
                              : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Create Lecture Modal */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Create New Lecture</h3>
              <button
                className="btn btn-outline btn-sm"
                onClick={() => setShowCreateModal(false)}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <form onSubmit={handleCreateLecture}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Lecture Title *</label>
                  <input
                    type="text"
                    className="form-input"
                    value={newLecture.title}
                    onChange={(e) => setNewLecture({ ...newLecture, title: e.target.value })}
                    placeholder="Introduction to Machine Learning"
                    required
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Description</label>
                  <textarea
                    className="form-input"
                    value={newLecture.description}
                    onChange={(e) => setNewLecture({ ...newLecture, description: e.target.value })}
                    placeholder="Brief description of the lecture content..."
                    rows="3"
                  />
                </div>
                <div className="grid grid-2">
                  <div className="form-group">
                    <label className="form-label">Course Code</label>
                    <input
                      type="text"
                      className="form-input"
                      value={newLecture.course_code}
                      onChange={(e) => setNewLecture({ ...newLecture, course_code: e.target.value })}
                      placeholder="CS101"
                    />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Course Name</label>
                    <input
                      type="text"
                      className="form-input"
                      value={newLecture.course_name}
                      onChange={(e) => setNewLecture({ ...newLecture, course_name: e.target.value })}
                      placeholder="Computer Science"
                    />
                  </div>
                </div>
              </div>
              <div className="modal-footer">
                <button
                  type="button"
                  className="btn btn-outline"
                  onClick={() => setShowCreateModal(false)}
                >
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create Lecture
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
