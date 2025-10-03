import { http } from './http';

export const getMe = (token) =>
  http.get('/users/me', { headers:{ Authorization:`Bearer ${token}` } });

export const updateMe = (payload, token) =>
  http.put('/users/me', payload, { headers:{ Authorization:`Bearer ${token}` } });

export const changePassword = (payload, token) =>
  http.put('/users/me/password', payload, { headers:{ Authorization:`Bearer ${token}` } });
