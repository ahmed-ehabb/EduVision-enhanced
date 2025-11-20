import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../hooks/useAuth'
import api from '../services/api'
import './Login.css'

export default function Login() {
  const navigate = useNavigate()
  const { login, user } = useAuth()

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    role: 'teacher',
  })
  const [showPassword, setShowPassword] = useState(false)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking')

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth()
  }, [])

  // Redirect if already logged in
  useEffect(() => {
    if (user) {
      const redirectPath = user.role === 'teacher' ? '/teacher/dashboard' : '/student/dashboard'
      navigate(redirectPath, { replace: true })
    }
  }, [user, navigate])

  const checkBackendHealth = async () => {
    try {
      const response = await api.get('/health')
      if (response.data.status === 'healthy') {
        setBackendStatus('connected')
      } else {
        setBackendStatus('error')
      }
    } catch (err) {
      setBackendStatus('disconnected')
    }
  }

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
    setError('')
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      await login(formData.email, formData.password)
      const redirectPath = formData.role === 'teacher' ? '/teacher/dashboard' : '/student/dashboard'
      navigate(redirectPath, { replace: true })
    } catch (err) {
      if (err.response?.status === 401) {
        setError('Invalid email or password')
      } else if (err.response?.status === 403) {
        setError('Account is inactive. Please contact support.')
      } else {
        setError('Unable to connect to server. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleDemoLogin = async (role) => {
    const demoCredentials = {
      teacher: {
        email: 'integration.teacher@eduvision.com',
        password: 'TeacherPass123!',
      },
      student: {
        email: 'integration.student@eduvision.com',
        password: 'StudentPass123!',
      },
    }

    const credentials = demoCredentials[role]
    setFormData({
      email: credentials.email,
      password: credentials.password,
      role: role,
    })

    setError('')
    setLoading(true)

    try {
      await login(credentials.email, credentials.password)
      const redirectPath = role === 'teacher' ? '/teacher/dashboard' : '/student/dashboard'
      navigate(redirectPath, { replace: true })
    } catch (err) {
      setError('Demo login failed. Please check if test accounts exist.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-container">
      {/* Left Side - Login Form */}
      <div className="login-form-side">
        <div className="login-form-wrapper">
          {/* Logo/Brand */}
          <div className="login-brand">
            <div className="login-logo">
              <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
                <rect width="40" height="40" rx="8" fill="#4F46E5" />
                <path
                  d="M20 10L28 14V20C28 25.5 24.5 30 20 32C15.5 30 12 25.5 12 20V14L20 10Z"
                  fill="white"
                />
              </svg>
            </div>
            <h1 className="login-title">EduVision</h1>
          </div>

          {/* Backend Status */}
          <div className={`backend-status backend-status-${backendStatus}`}>
            <span className="status-dot"></span>
            <span className="status-text">
              {backendStatus === 'connected' && 'Connected to server'}
              {backendStatus === 'disconnected' && 'Server offline'}
              {backendStatus === 'checking' && 'Checking connection...'}
              {backendStatus === 'error' && 'Connection error'}
            </span>
          </div>

          {/* Welcome Text */}
          <div className="login-welcome">
            <h2>Welcome Back</h2>
            <p>Sign in to continue to your dashboard</p>
          </div>

          {/* Error Alert */}
          {error && (
            <div className="alert alert-danger">
              {error}
            </div>
          )}

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="login-form">
            {/* Role Selection */}
            <div className="form-group">
              <label htmlFor="role" className="form-label">
                I am a
              </label>
              <select
                id="role"
                name="role"
                value={formData.role}
                onChange={handleChange}
                className="form-select"
                disabled={loading}
              >
                <option value="teacher">Teacher</option>
                <option value="student">Student</option>
                <option value="admin">Administrator</option>
              </select>
            </div>

            {/* Email Input */}
            <div className="form-group">
              <label htmlFor="email" className="form-label">
                Email Address
              </label>
              <input
                type="email"
                id="email"
                name="email"
                value={formData.email}
                onChange={handleChange}
                placeholder="you@example.com"
                className="form-input"
                required
                disabled={loading}
                autoComplete="email"
              />
            </div>

            {/* Password Input */}
            <div className="form-group">
              <label htmlFor="password" className="form-label">
                Password
              </label>
              <div className="password-input-wrapper">
                <input
                  type={showPassword ? 'text' : 'password'}
                  id="password"
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="Enter your password"
                  className="form-input"
                  required
                  disabled={loading}
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  className="password-toggle"
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={loading}
                >
                  {showPassword ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                      <line x1="1" y1="1" x2="23" y2="23" />
                    </svg>
                  ) : (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                      <circle cx="12" cy="12" r="3" />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {/* Remember Me & Forgot Password */}
            <div className="form-options">
              <label className="checkbox-label">
                <input type="checkbox" className="checkbox" />
                <span>Remember me</span>
              </label>
              <a href="#" className="forgot-password">
                Forgot password?
              </a>
            </div>

            {/* Submit Button */}
            <button type="submit" className="btn btn-primary btn-lg w-full" disabled={loading}>
              {loading ? (
                <>
                  <div className="spinner spinner-sm"></div>
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          {/* Demo Mode */}
          <div className="demo-section">
            <div className="demo-divider">
              <span>Quick Demo Access</span>
            </div>
            <div className="demo-buttons">
              <button
                type="button"
                className="btn btn-outline"
                onClick={() => handleDemoLogin('teacher')}
                disabled={loading}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                  <circle cx="9" cy="7" r="4" />
                  <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                  <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                </svg>
                Demo as Teacher
              </button>
              <button
                type="button"
                className="btn btn-outline"
                onClick={() => handleDemoLogin('student')}
                disabled={loading}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                  <circle cx="12" cy="7" r="4" />
                </svg>
                Demo as Student
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Right Side - Welcome/Info */}
      <div className="login-info-side">
        <div className="login-info-content">
          <div className="info-icon">
            <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M22 10v6M2 10l10-5 10 5-10 5z" />
              <path d="M6 12v5c0 1 2 2 6 2s6-1 6-2v-5" />
            </svg>
          </div>
          <h2>Educational Engagement Platform</h2>
          <p>
            Real-time student monitoring and analytics for enhanced learning experiences.
            Track engagement, analyze emotions, and improve educational outcomes.
          </p>

          <div className="feature-list">
            <div className="feature-item">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Real-time engagement monitoring</span>
            </div>
            <div className="feature-item">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Emotion and attention analysis</span>
            </div>
            <div className="feature-item">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Comprehensive session analytics</span>
            </div>
            <div className="feature-item">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              <span>Secure and privacy-focused</span>
            </div>
          </div>

          <div className="version-info">
            <span>Version 1.0.0</span>
            <span>•</span>
            <a href="#" className="info-link">Documentation</a>
            <span>•</span>
            <a href="#" className="info-link">Support</a>
          </div>
        </div>
      </div>
    </div>
  )
}
