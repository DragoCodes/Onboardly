# Load Test Results

Below is a table summarizing the results of a load test performed using Locust:

| Type   | Name                             | # Requests | # Fails | Median (ms) | 95%ile (ms) | 99%ile (ms) | Average (ms) | Min (ms) | Max (ms) | Average size (bytes) | Current RPS | Current Failures/s |
|--------|----------------------------------|------------|---------|-------------|-------------|-------------|--------------|----------|----------|----------------------|-------------|--------------------|
| POST   | /api/id/test-capture-and-compare | 4          | 0       | 42000       | 54000       | 54000       | 45247.48     | 41869    | 53693    | 18002                | 0           | 0                  |
| POST   | /api/id/test-upload-id          | 67         | 0       | 152000      | 310000      | 326000      | 156281.63    | 7438     | 326111   | 2833                 | 0.2         | 0                  |
| POST   | /api/id/verify-gesture-video    | 5          | 0       | 50000       | 60000       | 60000       | 53159.51     | 47966    | 59941    | 3061.6               | 0           | 0                  |
| DELETE | /api/sessions/cleanup           | 1          | 0       | 37378.53    | 37000       | 37000       | 37378.53     | 37379    | 37379    | 20                   | 0           | 0                  |
| POST   | /api/sessions/create-session    | 100        | 0       | 82000       | 87000       | 87000       | 75412.66     | 7        | 87368    | 53                   | 0           | 0                  |
|        | Aggregated                      | 177        | 0       | 83000       | 284000      | 321000      | 104498.87    | 7        | 326111   | 1595.75              | 0.2         | 0                  |

### Observations

- The table shows the performance of five distinct API endpoints, with four using the POST method and one using the DELETE method.
- The `/api/id/test-upload-id` endpoint has the highest number of requests (67) and the longest maximum response time (326111 ms).
- The `/api/sessions/create-session` endpoint has the highest number of requests overall (100) and the lowest minimum response time (7 ms).
- The aggregated row indicates a total of 177 requests across all endpoints, with no failures, an average response time of 104498.87 ms, and a current RPS of 0.2.
- The response sizes vary significantly, with the `/api/id/test-capture-and-compare` endpoint having the largest average response size (18002 bytes) and the `/api/sessions/cleanup` endpoint having the smallest (20 bytes).