import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import api from '../services/api'

export default function TeacherLectures() {
  const navigate = useNavigate()
  const [lectures, setLectures] = useState([])
  const [loading, setLoading] = useState(true)
  const [viewMode, setViewMode] = useState('grid') // 'grid' or 'list'

  useEffect(() => {
    fetchLectures()
  }, [])

  const fetchLectures = async () => {
    try {
      setLoading(true)
      const response = await api.get('/api/lectures')
      setLectures(response.data)
    } catch (error) {
      console.error('Failed to fetch lectures:', error)
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A'
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    })
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
          title="Lectures"
          subtitle="Manage your lecture content and AI-generated materials"
          actions={
            <button
              className="btn btn-primary"
              onClick={() => navigate('/teacher/lectures/upload')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload Lecture
            </button>
          }
        />

        <div className="dashboard-content">
          {/* View Toggle */}
          <div style={{ marginBottom: 'var(--spacing-lg)', display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
            <button
              className={`btn btn-sm ${viewMode === 'grid' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setViewMode('grid')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="7" height="7" />
                <rect x="14" y="3" width="7" height="7" />
                <rect x="14" y="14" width="7" height="7" />
                <rect x="3" y="14" width="7" height="7" />
              </svg>
            </button>
            <button
              className={`btn btn-sm ${viewMode === 'list' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setViewMode('list')}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="8" y1="6" x2="21" y2="6" />
                <line x1="8" y1="12" x2="21" y2="12" />
                <line x1="8" y1="18" x2="21" y2="18" />
                <line x1="3" y1="6" x2="3.01" y2="6" />
                <line x1="3" y1="12" x2="3.01" y2="12" />
                <line x1="3" y1="18" x2="3.01" y2="18" />
              </svg>
            </button>
          </div>

          {lectures.length === 0 ? (
            <div className="card">
              <div className="card-body">
                <div className="empty-state">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
                    <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
                  </svg>
                  <h4>No lectures yet</h4>
                  <p>Upload your first lecture to get started with AI-powered analysis</p>
                  <button
                    className="btn btn-primary"
                    onClick={() => navigate('/teacher/lectures/upload')}
                  >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <line x1="12" y1="5" x2="12" y2="19" />
                      <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    Upload Lecture
                  </button>
                </div>
              </div>
            </div>
          ) : viewMode === 'grid' ? (
            <div className="grid grid-3">
              {lectures.map((lecture) => (
                <div
                  key={lecture.lecture_id}
                  className="card"
                  style={{ cursor: 'pointer' }}
                  onClick={() => navigate(`/teacher/lectures/${lecture.lecture_id}`)}
                >
                  {lecture.thumbnail_url && (
                    <img
                      src={lecture.thumbnail_url}
                      alt={lecture.title}
                      style={{ width: '100%', height: '180px', objectFit: 'cover', borderRadius: 'var(--radius-lg) var(--radius-lg) 0 0' }}
                    />
                  )}
                  <div className="card-body">
                    <h4 style={{ marginBottom: '0.5rem' }}>{lecture.title}</h4>
                    {lecture.course_code && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                        {lecture.course_code} - {lecture.course_name}
                      </div>
                    )}
                    {lecture.description && (
                      <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginBottom: '1rem', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                        {lecture.description}
                      </p>
                    )}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: 'var(--text-light)' }}>
                      <span>{formatDate(lecture.created_at)}</span>
                      <span className={`status ${lecture.is_published ? 'status-active' : 'status-inactive'}`}>
                        {lecture.is_published ? 'Published' : 'Draft'}
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="card">
              <div className="table-container">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Title</th>
                      <th>Course</th>
                      <th>Created</th>
                      <th>Duration</th>
                      <th>Status</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lectures.map((lecture) => (
                      <tr key={lecture.lecture_id}>
                        <td>
                          <div style={{ fontWeight: 600 }}>{lecture.title}</div>
                          {lecture.description && (
                            <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                              {lecture.description.substring(0, 60)}...
                            </div>
                          )}
                        </td>
                        <td>
                          {lecture.course_code && (
                            <div>
                              <div style={{ fontWeight: 500 }}>{lecture.course_code}</div>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>{lecture.course_name}</div>
                            </div>
                          )}
                        </td>
                        <td>{formatDate(lecture.created_at)}</td>
                        <td>{lecture.video_duration ? `${Math.floor(lecture.video_duration / 60)} min` : 'N/A'}</td>
                        <td>
                          <span className={`status ${lecture.is_published ? 'status-active' : 'status-inactive'}`}>
                            {lecture.is_published ? 'Published' : 'Draft'}
                          </span>
                        </td>
                        <td>
                          <button
                            className="btn btn-sm btn-outline"
                            onClick={(e) => {
                              e.stopPropagation()
                              navigate(`/teacher/lectures/${lecture.lecture_id}`)
                            }}
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
