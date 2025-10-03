import axios from 'axios';
import { API_URL } from '../config';

export const http = axios.create({
  baseURL: `${API_URL}/api`,
  withCredentials: false
});

export const setAuthToken = (token) => {
  if (token) {
    http.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  } else {
    delete http.defaults.headers.common['Authorization'];
  }
};
