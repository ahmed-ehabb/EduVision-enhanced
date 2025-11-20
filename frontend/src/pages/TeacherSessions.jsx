import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import { sessionService } from '../services/sessionService'
import api from '../services/api'

export default function TeacherSessions() {
  const navigate = useNavigate()
  const [sessions, setSessions] = useState([])
  const [lectures, setLectures] = useState([])
  const [loading, setLoading] = useState(true)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedLecture, setSelectedLecture] = useState('')

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setLoading(true)
      const [sessionsRes, lecturesRes] = await Promise.all([
        api.get('/api/sessions'),
        api.get('/api/lectures')
      ])
      setSessions(sessionsRes.data)
      setLectures(lecturesRes.data)
    } catch (error) {
      console.error('Failed to fetch data:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateSession = async (e) => {
    e.preventDefault()
    if (!selectedLecture) {
      alert('Please select a lecture')
      return
    }

    try {
      await sessionService.createSession(selectedLecture)
      setShowCreateModal(false)
      setSelectedLecture('')
      fetchData()
    } catch (error) {
      console.error('Failed to create session:', error)
      alert('Failed to create session. Please try again.')
    }
  }

  const handleStartSession = async (sessionCode) => {
    if (!confirm('Start this session?')) return

    try {
      await sessionService.startSession(sessionCode)
      fetchData()
    } catch (error) {
      console.error('Failed to start session:', error)
      alert('Failed to start session.')
    }
  }

  const handleEndSession = async (sessionCode) => {
    if (!confirm('End this session?')) return

    try {
      await sessionService.endSession(sessionCode)
      fetchData()
    } catch (error) {
      console.error('Failed to end session:', error)
      alert('Failed to end session.')
    }
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleString()
  }

  const getStatusClass = (status) => {
    switch (status) {
      case 'active': return 'status-active'
      case 'completed': return 'status-completed'
      case 'scheduled': return 'status-pending'
      default: return 'status-inactive'
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
          title="Sessions"
          subtitle="Manage your lecture sessions"
          actions={
            <button className="btn btn-primary" onClick={() => setShowCreateModal(true)}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              Create Session
            </button>
          }
        />

        <div className="dashboard-content">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">All Sessions</h3>
              <p className="card-subtitle">{sessions.length} total sessions</p>
            </div>
            <div className="card-body">
              {sessions.length === 0 ? (
                <div className="empty-state">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M12 2L2 7l10 5 10-5-10-5z" />
                    <path d="M2 17l10 5 10-5" />
                    <path d="M2 12l10 5 10-5" />
                  </svg>
                  <h4>No sessions yet</h4>
                  <p>Create a session to get started</p>
                  <button className="btn btn-primary btn-sm" onClick={() => setShowCreateModal(true)}>
                    Create Session
                  </button>
                </div>
              ) : (
                <div className="table-container">
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Session Code</th>
                        <th>Status</th>
                        <th>Created</th>
                        <th>Started</th>
                        <th>Duration</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sessions.map((session) => (
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
                          <td>{formatDate(session.scheduled_at)}</td>
                          <td>{formatDate(session.started_at)}</td>
                          <td>{session.duration || 'N/A'}</td>
                          <td>
                            <div style={{ display: 'flex', gap: '0.5rem' }}>
                              {session.status === 'scheduled' && (
                                <button
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => handleStartSession(session.session_code)}
                                >
                                  Start
                                </button>
                              )}
                              {session.status === 'active' && (
                                <button
                                  className="btn btn-danger btn-sm"
                                  onClick={() => handleEndSession(session.session_code)}
                                >
                                  End
                                </button>
                              )}
                            </div>
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

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Create New Session</h3>
              <button className="btn btn-outline btn-sm" onClick={() => setShowCreateModal(false)}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
            <form onSubmit={handleCreateSession}>
              <div className="modal-body">
                <div className="form-group">
                  <label className="form-label">Select Lecture *</label>
                  <select
                    className="form-select"
                    value={selectedLecture}
                    onChange={(e) => setSelectedLecture(e.target.value)}
                    required
                  >
                    <option value="">Choose a lecture...</option>
                    {lectures.map((lecture) => (
                      <option key={lecture.lecture_id} value={lecture.lecture_id}>
                        {lecture.title}
                      </option>
                    ))}
                  </select>
                  <p className="form-help">
                    A session code will be automatically generated for students to join
                  </p>
                </div>
              </div>
              <div className="modal-footer">
                <button type="button" className="btn btn-outline" onClick={() => setShowCreateModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary">
                  Create Session
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
