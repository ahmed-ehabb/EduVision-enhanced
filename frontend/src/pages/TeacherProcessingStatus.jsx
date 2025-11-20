import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import lectureService from '../services/lectureService'
import toast from 'react-hot-toast'

export default function TeacherProcessingStatus() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    let interval

    const checkStatus = async () => {
      try {
        const data = await lectureService.getProcessingStatus(jobId)
        setStatus(data)
        setLoading(false)

        // If processing is complete, navigate to lecture detail after a brief delay
        if (data.status === 'completed') {
          clearInterval(interval)
          toast.success('Lecture processing completed!')
          setTimeout(() => {
            navigate(`/teacher/lectures/${jobId}`)
          }, 2000)
        } else if (data.status === 'failed') {
          clearInterval(interval)
          setError(data.error || 'Processing failed')
          toast.error('Lecture processing failed')
        }
      } catch (err) {
        console.error('Failed to fetch status:', err)
        setError(err.message)
        setLoading(false)
        clearInterval(interval)
      }
    }

    // Initial check
    checkStatus()

    // Poll every 3 seconds
    interval = setInterval(checkStatus, 3000)

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [jobId, navigate])

  const getProgressPercentage = () => {
    if (!status || !status.progress) return 0

    const steps = [
      'transcription',
      'translation',
      'notes_generation',
      'quiz_generation',
      'content_alignment'
    ]

    const completedSteps = steps.filter(step => status.progress[step] === 'completed').length
    return Math.round((completedSteps / steps.length) * 100)
  }

  const getStepStatus = (step) => {
    if (!status || !status.progress) return 'pending'
    return status.progress[step] || 'pending'
  }

  const getStepIcon = (stepStatus) => {
    if (stepStatus === 'completed') {
      return (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      )
    } else if (stepStatus === 'processing') {
      return <div className="spinner spinner-sm"></div>
    } else if (stepStatus === 'failed') {
      return (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
      )
    } else {
      return (
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
        </svg>
      )
    }
  }

  const steps = [
    { key: 'transcription', label: 'Speech Recognition', description: 'Converting audio to text' },
    { key: 'translation', label: 'Translation', description: 'Translating to English (if needed)' },
    { key: 'notes_generation', label: 'Notes Generation', description: 'Generating AI lecture notes' },
    { key: 'quiz_generation', label: 'Quiz Generation', description: 'Creating quiz questions' },
    { key: 'content_alignment', label: 'Content Alignment', description: 'Analyzing alignment with textbook' },
  ]

  if (loading && !status) {
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
          title="Processing Lecture"
          subtitle="AI processing in progress"
          actions={
            status?.status === 'completed' && (
              <button
                className="btn btn-primary"
                onClick={() => navigate(`/teacher/lectures/${jobId}`)}
              >
                View Lecture
              </button>
            )
          }
        />

        <div className="dashboard-content">
          {error ? (
            <div className="card">
              <div className="card-body">
                <div className="empty-state">
                  <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" style={{ color: 'var(--color-danger)' }}>
                    <circle cx="12" cy="12" r="10" />
                    <line x1="15" y1="9" x2="9" y2="15" />
                    <line x1="9" y1="9" x2="15" y2="15" />
                  </svg>
                  <h4>Processing Failed</h4>
                  <p>{error}</p>
                  <button
                    className="btn btn-primary"
                    onClick={() => navigate('/teacher/lectures/upload')}
                  >
                    Upload Another Lecture
                  </button>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Overall Progress */}
              <div className="card">
                <div className="card-body">
                  <div style={{ marginBottom: 'var(--spacing-lg)' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 'var(--spacing-sm)' }}>
                      <h3>Overall Progress</h3>
                      <span style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--color-primary)' }}>
                        {getProgressPercentage()}%
                      </span>
                    </div>
                    <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--color-light-gray)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${getProgressPercentage()}%`,
                          height: '100%',
                          backgroundColor: 'var(--color-primary)',
                          transition: 'width 0.3s ease',
                        }}
                      />
                    </div>
                  </div>

                  {status?.status === 'processing' && (
                    <div style={{ padding: 'var(--spacing-md)', backgroundColor: 'var(--color-primary-light)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                      <p style={{ margin: 0, color: 'var(--color-primary)', fontWeight: 500 }}>
                        AI is processing your lecture. This may take a few minutes...
                      </p>
                    </div>
                  )}

                  {status?.status === 'completed' && (
                    <div style={{ padding: 'var(--spacing-md)', backgroundColor: 'var(--color-success-light)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                      <p style={{ margin: 0, color: 'var(--color-success)', fontWeight: 500 }}>
                        Processing complete! Redirecting to lecture details...
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Processing Steps */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Processing Steps</h3>
                </div>
                <div className="card-body">
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
                    {steps.map((step, index) => {
                      const stepStatus = getStepStatus(step.key)
                      const isActive = stepStatus === 'processing'
                      const isCompleted = stepStatus === 'completed'
                      const isFailed = stepStatus === 'failed'

                      return (
                        <div
                          key={step.key}
                          style={{
                            display: 'flex',
                            gap: 'var(--spacing-md)',
                            padding: 'var(--spacing-md)',
                            backgroundColor: isActive ? 'var(--color-primary-light)' : 'transparent',
                            borderRadius: 'var(--radius-md)',
                            transition: 'all var(--transition-fast)',
                          }}
                        >
                          <div
                            style={{
                              flexShrink: 0,
                              width: '48px',
                              height: '48px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              borderRadius: '50%',
                              backgroundColor: isCompleted
                                ? 'var(--color-success)'
                                : isFailed
                                ? 'var(--color-danger)'
                                : isActive
                                ? 'var(--color-primary)'
                                : 'var(--color-light-gray)',
                              color: isCompleted || isFailed || isActive ? 'white' : 'var(--text-secondary)',
                            }}
                          >
                            {getStepIcon(stepStatus)}
                          </div>
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>{step.label}</h4>
                              <span
                                className={`status ${
                                  isCompleted
                                    ? 'status-active'
                                    : isFailed
                                    ? 'status-inactive'
                                    : isActive
                                    ? 'status-warning'
                                    : ''
                                }`}
                              >
                                {stepStatus === 'completed'
                                  ? 'Completed'
                                  : stepStatus === 'processing'
                                  ? 'Processing...'
                                  : stepStatus === 'failed'
                                  ? 'Failed'
                                  : 'Pending'}
                              </span>
                            </div>
                            <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                              {step.description}
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </div>
              </div>

              {/* Metadata (if available) */}
              {status?.result && (
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Lecture Information</h3>
                  </div>
                  <div className="card-body">
                    <div className="grid grid-2">
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          Title
                        </div>
                        <div style={{ fontWeight: 600 }}>{status.result.title || 'N/A'}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          Duration
                        </div>
                        <div style={{ fontWeight: 600 }}>
                          {status.result.duration ? `${Math.floor(status.result.duration / 60)} min` : 'N/A'}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          Language
                        </div>
                        <div style={{ fontWeight: 600 }}>{status.result.language || 'N/A'}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                          Course
                        </div>
                        <div style={{ fontWeight: 600 }}>{status.result.course_code || 'N/A'}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
