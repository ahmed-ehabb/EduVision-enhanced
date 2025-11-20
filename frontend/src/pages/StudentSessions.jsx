import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'

export default function StudentSessions() {
  return (
    <div className="dashboard-layout">
      <Sidebar role="student" />
      <div className="dashboard-main">
        <Header title="My Sessions" subtitle="View your session history" />
        <div className="dashboard-content">
          <div className="card">
            <div className="card-body">
              <div className="empty-state">
                <h4>Session History</h4>
                <p>Your session history will appear here.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
