

# Add the following line to your crontab (edit with `crontab -e`)
# This will run the script every 15 minutes
*/5 * * * * /bin/bash /mnt/disk6/heyuan/cron.sh >> /mnt/disk6/heyuan/cron.log 2>&1

crontab -l

# To remove the cron job, use:
crontab -r


#!/bin/bash
export no_proxy="localhost,10.239.15.41,127.0.0.1,::1"

# Check if hl-smi is installed
if ! command -v hl-smi &> /dev/null
then
    echo "hl-smi could not be found. Please install it first."
    exit 1
fi

# Run hl-smi and display output
hl-smi

# Optionally, extract and summarize usage info for each device
echo ""
echo "Summary of HL-225D Usage:"
hl_smi_output=$(hl-smi)
echo "$hl_smi_output" | awk '/HL-225D/ {dev=$2} /[0-9]+W \/ *[0-9]+W/ {print "AIP "dev": Power " $5 " / " $7 ", Memory " $11 " / " $13 ", Util " $15}'

# Check if all usage values are 0
usage_values=$(echo "$hl_smi_output" | awk '/[0-9]+W \/ *[0-9]+W/ {print $(NF-2)}')
all_zero=true
for usage in $usage_values; do
  # Remove % sign if present
  usage_num=${usage%%%}
  echo "Device usage: $usage_num"
  if [ "$usage_num" != "0" ]; then
    all_zero=false
    break
  fi
done

if $all_zero; then
  echo "All usage is 0, executing command..."
  cd /mnt/disk6/heyuan
  curl -X POST "http://10.239.15.41:9396/v1/videos" \
    -H "Content-Type: multipart/form-data" \
    -F "input_reference=@test.png" \
    -F "audio=@test.wav"
fi