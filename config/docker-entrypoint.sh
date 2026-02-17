#!/bin/bash
# Process environment variables in config file
envsubst < /freqtrade/config/freqtrade_config.json > /tmp/config.json

# Start freqtrade with processed config
exec freqtrade trade \
    --config /tmp/config.json \
    --strategy EMARSIStrategy \
    --strategy-path /freqtrade/user_data/strategies
