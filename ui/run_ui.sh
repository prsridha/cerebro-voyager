#!/bin/bash

echo "
export const environment = {
    backendURL: 'http://$BACKEND_HOST:30083'
};
" | tee /cerebro-core/ui/project-cerebro/src/environments/environment.ts /cerebro-core/ui/project-cerebro/src/environments/environment.development.ts

(cd /cerebro-core/ui/project-cerebro && npm install)
(export NG_CLI_ANALYTICS="false" && cd /cerebro-core/ui/project-cerebro && ng serve --host 0.0.0.0 --port 80 --disable-host-check)
# sleep infinity
