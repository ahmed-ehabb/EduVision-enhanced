import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'

export default function TeacherAnalytics() {
  return (
    <div className="dashboard-layout">
      <Sidebar role="teacher" />
      <div className="dashboard-main">
        <Header title="Analytics" subtitle="Session analytics and insights" />
        <div className="dashboard-content">
          <div className="card">
            <div className="card-body">
              <div className="empty-state">
                <h4>Analytics Dashboard</h4>
                <p>Analytics and reporting features coming soon.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
