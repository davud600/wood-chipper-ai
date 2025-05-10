#!/bin/bash

set -euxo pipefail

cat <<EOF > ~/../etc/nginx/nginx.conf
events { worker_connections 2048; }
http {
    client_max_body_size 1024M;

    # wood chipper ai server
    server {
        listen 8000;

        location / {
            add_header Cache-Control no-cache;
            proxy_pass http://localhost:8001;
            proxy_intercept_errors on;
            error_page 502 =200 @502;
        }

        location /README.md{
            root /usr/share/nginx/html;
        }

        location @502 {
            add_header Cache-Control no-cache;
            root /usr/share/nginx/html;
            rewrite ^(.*)$ /readme.html break;
        }
    }
}
EOF

nginx -s reload || nginx

chmod +x /workspace/wood-chipper-ai/start.sh

~/../workspace/wood-chipper-ai/start.sh
