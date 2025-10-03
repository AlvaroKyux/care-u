import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import Signup from './pages/Signup';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import CreateIncident from './pages/CreateIncident';
import Feed from './pages/Feed';

function PrivateRoute({ children }){
  const { token } = useAuth();
  return token ? children : <Navigate to="/login" replace />;
}

export default function App(){
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="/signup" element={<Signup/>} />
          <Route path="/login" element={<Login/>} />
          <Route path="/dashboard" element={<PrivateRoute><Dashboard/></PrivateRoute>} />
          <Route path="*" element={<Navigate to="/login" replace />} />
          <Route path="/feed" element={<PrivateRoute><Feed/></PrivateRoute>} />
          <Route path="/new" element={<PrivateRoute><CreateIncident/></PrivateRoute>} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
