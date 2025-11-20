import { NavLink } from 'react-router-dom'
import { useAuth } from '../../hooks/useAuth'
import './Sidebar.css'

export default function Sidebar({ role }) {
  const { user, logout } = useAuth()

  const teacherMenuItems = [
    {
      path: '/teacher/dashboard',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="7" height="7" />
          <rect x="14" y="3" width="7" height="7" />
          <rect x="14" y="14" width="7" height="7" />
          <rect x="3" y="14" width="7" height="7" />
        </svg>
      ),
      label: 'Dashboard',
    },
    {
      path: '/teacher/sessions',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      ),
      label: 'Sessions',
    },
    {
      path: '/teacher/lectures',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
        </svg>
      ),
      label: 'Lectures',
    },
    {
      path: '/teacher/analytics',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="18" y1="20" x2="18" y2="10" />
          <line x1="12" y1="20" x2="12" y2="4" />
          <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
      ),
      label: 'Analytics',
    },
    {
      path: '/teacher/students',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
          <circle cx="9" cy="7" r="4" />
          <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
          <path d="M16 3.13a4 4 0 0 1 0 7.75" />
        </svg>
      ),
      label: 'Students',
    },
    {
      path: '/teacher/test',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 11l3 3L22 4" />
          <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
        </svg>
      ),
      label: 'System Test',
    },
  ]

  const studentMenuItems = [
    {
      path: '/student/dashboard',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="7" height="7" />
          <rect x="14" y="3" width="7" height="7" />
          <rect x="14" y="14" width="7" height="7" />
          <rect x="3" y="14" width="7" height="7" />
        </svg>
      ),
      label: 'Dashboard',
    },
    {
      path: '/student/sessions',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M12 2L2 7l10 5 10-5-10-5z" />
          <path d="M2 17l10 5 10-5" />
          <path d="M2 12l10 5 10-5" />
        </svg>
      ),
      label: 'My Sessions',
    },
    {
      path: '/student/stats',
      icon: (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="18" y1="20" x2="18" y2="10" />
          <line x1="12" y1="20" x2="12" y2="4" />
          <line x1="6" y1="20" x2="6" y2="14" />
        </svg>
      ),
      label: 'My Stats',
    },
  ]

  const menuItems = role === 'teacher' ? teacherMenuItems : studentMenuItems

  return (
    <aside className="sidebar">
      {/* Logo/Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <svg width="32" height="32" viewBox="0 0 40 40" fill="none">
            <rect width="40" height="40" rx="8" fill="#4F46E5" />
            <path
              d="M20 10L28 14V20C28 25.5 24.5 30 20 32C15.5 30 12 25.5 12 20V14L20 10Z"
              fill="white"
            />
          </svg>
        </div>
        <span className="sidebar-brand-text">EduVision</span>
      </div>

      {/* User Info */}
      <div className="sidebar-user">
        <div className="sidebar-user-avatar">
          {user?.name?.charAt(0).toUpperCase() || user?.email?.charAt(0).toUpperCase() || 'U'}
        </div>
        <div className="sidebar-user-info">
          <div className="sidebar-user-name">{user?.name || user?.email}</div>
          <div className="sidebar-user-role">
            {role === 'teacher' ? 'Teacher' : 'Student'}
          </div>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="sidebar-nav">
        <ul className="sidebar-menu">
          {menuItems.map((item) => (
            <li key={item.path}>
              <NavLink
                to={item.path}
                className={({ isActive }) =>
                  `sidebar-link ${isActive ? 'sidebar-link-active' : ''}`
                }
              >
                <span className="sidebar-link-icon">{item.icon}</span>
                <span className="sidebar-link-label">{item.label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Bottom Actions */}
      <div className="sidebar-footer">
        <button className="sidebar-link" onClick={logout}>
          <span className="sidebar-link-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
          </span>
          <span className="sidebar-link-label">Logout</span>
        </button>
      </div>
    </aside>
  )
}
