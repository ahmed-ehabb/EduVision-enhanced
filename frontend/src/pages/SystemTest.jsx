import { useState } from 'react'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import api from '../services/api'
import lectureService from '../services/lectureService'
import toast from 'react-hot-toast'

export default function SystemTest() {
  const [testResults, setTestResults] = useState({})
  const [testing, setTesting] = useState(false)

  const addResult = (test, success, message, data = null) => {
    setTestResults(prev => ({
      ...prev,
      [test]: { success, message, data, timestamp: new Date().toISOString() }
    }))
  }

  const runAllTests = async () => {
    setTesting(true)
    setTestResults({})

    // Test 1: Unified API Health
    try {
      const response = await api.get('/health')
      addResult('unifiedAPI', true, 'Unified API (Port 8002) is responding', response.data)
    } catch (error) {
      addResult('unifiedAPI', false, `Unified API failed: ${error.message}`)
    }

    // Test 2: Teacher Module API Health
    try {
      const response = await lectureService.checkHealth()
      addResult('teacherAPI', true, 'Teacher Module API (Port 8000) is responding', response)
    } catch (error) {
      addResult('teacherAPI', false, `Teacher Module API failed: ${error.message}`)
    }

    // Test 3: Fetch User Profile
    try {
      const response = await api.get('/auth/me')
      addResult('profile', true, 'Successfully fetched user profile', response.data)
    } catch (error) {
      addResult('profile', false, `Profile fetch failed: ${error.message}`)
    }

    // Test 4: Fetch Lectures
    try {
      const response = await api.get('/api/lectures')
      addResult('lectures', true, `Found ${response.data.length} lectures`, response.data)
    } catch (error) {
      addResult('lectures', false, `Lectures fetch failed: ${error.message}`)
    }

    // Test 5: Fetch Sessions
    try {
      const response = await api.get('/api/sessions')
      addResult('sessions', true, `Found ${response.data.length} sessions`, response.data)
    } catch (error) {
      addResult('sessions', false, `Sessions fetch failed: ${error.message}`)
    }

    setTesting(false)
  }

  const createTestLecture = async () => {
    try {
      const response = await api.post('/api/lectures', {
        title: 'Test Lecture - ' + new Date().toLocaleTimeString(),
        description: 'This is a test lecture created from the system test page',
        course_code: 'TEST101',
        course_name: 'Testing Course'
      })
      toast.success('Test lecture created successfully!')
      addResult('createLecture', true, 'Lecture created', response.data)

      // Refresh lectures list
      const lecturesRes = await api.get('/api/lectures')
      addResult('lectures', true, `Found ${lecturesRes.data.length} lectures`, lecturesRes.data)
    } catch (error) {
      toast.error('Failed to create lecture')
      addResult('createLecture', false, error.message)
    }
  }

  const getStatusIcon = (success) => {
    if (success) {
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: '#22c55e' }}>
          <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
          <polyline points="22 4 12 14.01 9 11.01" />
        </svg>
      )
    } else {
      return (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" style={{ color: '#ef4444' }}>
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
      )
    }
  }

  return (
    <div className="dashboard-layout">
      <Sidebar role="teacher" />
      <div className="dashboard-main">
        <Header
          title="System Integration Test"
          subtitle="Test backend API connectivity and functionality"
          actions={
            <>
              <button
                className="btn btn-outline"
                onClick={runAllTests}
                disabled={testing}
              >
                {testing ? (
                  <>
                    <div className="spinner spinner-sm"></div>
                    Testing...
                  </>
                ) : (
                  'Run All Tests'
                )}
              </button>
              <button
                className="btn btn-primary"
                onClick={createTestLecture}
              >
                Create Test Lecture
              </button>
            </>
          }
        />

        <div className="dashboard-content">
          {/* API Status Cards */}
          <div className="grid grid-2">
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Unified API</h3>
                <p className="card-subtitle">Port 8002 - Auth, Lectures, Sessions</p>
              </div>
              <div className="card-body">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <strong>Base URL:</strong>
                  <code style={{ padding: '0.25rem 0.5rem', backgroundColor: 'var(--color-light-gray)', borderRadius: '4px', fontSize: '0.875rem' }}>
                    http://localhost:8002
                  </code>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <strong>Status:</strong>
                  {testResults.unifiedAPI ? (
                    <span style={{ color: testResults.unifiedAPI.success ? '#22c55e' : '#ef4444' }}>
                      {testResults.unifiedAPI.message}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-secondary)' }}>Not tested</span>
                  )}
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Teacher Module API</h3>
                <p className="card-subtitle">Port 8001 - AI Processing, File Upload</p>
              </div>
              <div className="card-body">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <strong>Base URL:</strong>
                  <code style={{ padding: '0.25rem 0.5rem', backgroundColor: 'var(--color-light-gray)', borderRadius: '4px', fontSize: '0.875rem' }}>
                    http://localhost:8001
                  </code>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <strong>Status:</strong>
                  {testResults.teacherAPI ? (
                    <span style={{ color: testResults.teacherAPI.success ? '#22c55e' : '#ef4444' }}>
                      {testResults.teacherAPI.message}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-secondary)' }}>Not tested</span>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Test Results */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Test Results</h3>
              <p className="card-subtitle">API endpoint connectivity and response status</p>
            </div>
            <div className="card-body">
              {Object.keys(testResults).length === 0 ? (
                <div className="empty-state">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <path d="M9 11l3 3L22 4" />
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
                  </svg>
                  <h4>No tests run yet</h4>
                  <p>Click "Run All Tests" to check system connectivity</p>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-md)' }}>
                  {Object.entries(testResults).map(([testName, result]) => (
                    <div
                      key={testName}
                      style={{
                        padding: 'var(--spacing-md)',
                        border: `1px solid ${result.success ? '#22c55e' : '#ef4444'}`,
                        borderRadius: 'var(--radius-md)',
                        backgroundColor: result.success ? 'rgba(34, 197, 94, 0.05)' : 'rgba(239, 68, 68, 0.05)'
                      }}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-md)', marginBottom: 'var(--spacing-sm)' }}>
                        {getStatusIcon(result.success)}
                        <div>
                          <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>
                            {testName.replace(/([A-Z])/g, ' $1').trim().toUpperCase()}
                          </h4>
                          <p style={{ margin: '0.25rem 0 0', fontSize: '0.875rem', color: 'var(--text-secondary)' }}>
                            {result.message}
                          </p>
                        </div>
                      </div>

                      {result.data && (
                        <details style={{ marginTop: 'var(--spacing-sm)' }}>
                          <summary style={{ cursor: 'pointer', fontSize: '0.875rem', fontWeight: 500, color: 'var(--color-primary)' }}>
                            View Response Data
                          </summary>
                          <pre style={{
                            marginTop: 'var(--spacing-sm)',
                            padding: 'var(--spacing-md)',
                            backgroundColor: 'var(--color-light-gray)',
                            borderRadius: 'var(--radius-md)',
                            fontSize: '0.75rem',
                            overflow: 'auto',
                            maxHeight: '300px'
                          }}>
                            {JSON.stringify(result.data, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Quick Actions</h3>
              <p className="card-subtitle">Common tasks for testing the system</p>
            </div>
            <div className="card-body">
              <div className="grid grid-3">
                <button
                  className="btn btn-outline"
                  onClick={() => window.location.href = '/teacher/lectures'}
                >
                  View Lectures
                </button>
                <button
                  className="btn btn-outline"
                  onClick={() => window.location.href = '/teacher/lectures/upload'}
                >
                  Upload Lecture
                </button>
                <button
                  className="btn btn-outline"
                  onClick={() => window.location.href = '/teacher/sessions'}
                >
                  View Sessions
                </button>
              </div>
            </div>
          </div>

          {/* System Information */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">System Information</h3>
            </div>
            <div className="card-body">
              <div className="grid grid-2">
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                    Authentication Token
                  </div>
                  <code style={{
                    display: 'block',
                    padding: '0.5rem',
                    backgroundColor: 'var(--color-light-gray)',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    wordBreak: 'break-all'
                  }}>
                    {localStorage.getItem('access_token') ?
                      localStorage.getItem('access_token').substring(0, 50) + '...' :
                      'Not logged in'}
                  </code>
                </div>
                <div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                    User Role
                  </div>
                  <code style={{
                    display: 'block',
                    padding: '0.5rem',
                    backgroundColor: 'var(--color-light-gray)',
                    borderRadius: '4px',
                    fontSize: '0.75rem'
                  }}>
                    {localStorage.getItem('user_role') || 'Unknown'}
                  </code>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
