using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Unity.LiveCapture;

namespace Movella.Xsens
{
    class XsensClient
    {
        public bool IsStarted => m_Thread is { IsAlive: true };

        public FrameRate FrameRate { get; set; } = StandardFrameRate.FPS_24_00;

        public bool IsConnected { get; private set; }
        public int Port { get; private set; }

        public event Func<int, FrameData, bool> FrameDataReceivedAsync;
        public event Action Disconnected;

        enum StreamingProtocol
        {
            SPPoseEuler = 1,
            SPPoseQuaternion = 2,
            SPPosePositions = 3,
            SPTagPositionsLegacy = 4,
            SPPoseUnity3D = 5,
            SPMetaScalingLegacy = 10,
            SPMetaPropInfoLegacy = 11,
            SPMetaMoreMeta = 12,
            SPMetaScaling = 13,
            SPTimecode = 25,
            SPLinearSegment = 21,
            SPCOM = 24
        };

        Thread m_Thread;
        UdpClient m_Client;
        (Timecode tc, FrameRate rate) m_Timecode;

        FrameData[] m_Frames = new FrameData[XsensConstants.NumStreams];
        object[] m_FrameMutexes = new object[XsensConstants.NumStreams];

        private int packetType;

        public XsensClient()
        {
            for (int i = 0; i < XsensConstants.NumStreams; i++)
                m_FrameMutexes[i] = new object();
        }

        public void Connect(int port)
        {
            if (port <= 0 || port > 0xFFFF)
            {
                Debug.LogError($"{nameof(XsensClient)}: Tried to connect with an invalid port: {port}");
                return;
            }

            if (!IsConnected)
            {
                try
                {
                    m_Thread = new Thread(() => ListenForDataAsync(port));
                    m_Thread.IsBackground = true;
                    m_Thread.Start();
                }
                catch (Exception e)
                {
                    m_Thread = null;
                    IsConnected = false;
                    Debug.Log($"{nameof(XsensClient)}({port}): Thread start exception {e}");
                }
            }
        }

        public void Disconnect()
        {
            IsConnected = false;

            if (m_Thread != null)
            {
                if (!m_Thread.Join(2000))
                    m_Thread.Abort();
                m_Thread = null;
            }

            m_Client?.Close();
            m_Client = null;

            Disconnected?.Invoke();
        }

        public FrameData GetFrame(int characterID)
        {
            if (IsConnected &&
                characterID >= 0 &&
                characterID < XsensConstants.NumStreams)
            {
                lock (m_FrameMutexes[characterID])
                    return m_Frames[characterID];
            }
            return default;
        }

        void ListenForDataAsync(int port)
        {
            try
            {
                Port = port;

                IPEndPoint endPoint = new IPEndPoint(IPAddress.Any, port);

                m_Client = new UdpClient(port);

                IsConnected = true;

                while (IsConnected)
                {
                    var receiveBytes = m_Client.Receive(ref endPoint);

                    string result = "";
                    result += (char)receiveBytes[4];
                    result += (char)receiveBytes[5];
                    StreamingProtocol packId = (StreamingProtocol)int.Parse(result);
                    packetType = (int)packId;

                    // Check this manual for more streaming information: https://www.xsens.com/hubfs/Downloads/Manuals/MVN_real-time_network_streaming_protocol_specification.pdf.

                    switch (packId)
                    {
                        // Data Packet Type 2: Segment Data (Position +) Quaternion.
                        case StreamingProtocol.SPPoseQuaternion:
                        {
                            if (receiveBytes.Length > 15)
                            {
                                int characterID = receiveBytes[16];

                                if (characterID >= 0 && characterID < XsensConstants.NumStreams)
                                {
                                    var frame = ParsePacket(receiveBytes);

                                    if (!(FrameDataReceivedAsync?.Invoke(characterID, frame) ?? false))
                                    {
                                        lock (m_FrameMutexes[characterID])
                                            m_Frames[characterID] = frame;
                                    }
                                }
                            }
                            break;
                        }

                        // Data Packet Type 21: Linear Segment Kinematics (Position + Global Velocity + Global Acceleration).
                        case StreamingProtocol.SPLinearSegment:
                        {
                            if (receiveBytes.Length > 15)
                            {
                                int characterID = receiveBytes[16];

                                if (characterID >= 0 && characterID < XsensConstants.NumStreams)
                                {
                                    var frame = ParsePacket(receiveBytes);

                                    if (!(FrameDataReceivedAsync?.Invoke(characterID, frame) ?? false))
                                    {
                                        lock (m_FrameMutexes[characterID])
                                            m_Frames[characterID] = frame;
                                    }
                                }
                            }
                            break;
                        }

                        // Data Packet Type 24: Center of Mass (Position + Global Velocity + Global Acceleration).
                        case StreamingProtocol.SPCOM:
                        {
                            if (receiveBytes.Length > 15)
                            {
                                int characterID = receiveBytes[16];

                                if (characterID >= 0 && characterID < XsensConstants.NumStreams)
                                {
                                    var frame = ParsePacket(receiveBytes);

                                    if (!(FrameDataReceivedAsync?.Invoke(characterID, frame) ?? false))
                                    {
                                        lock (m_FrameMutexes[characterID])
                                            m_Frames[characterID] = frame;
                                    }
                                }
                            }
                            break;
                        }

                        // Data Packet Type 25: Time Code.
                        case StreamingProtocol.SPTimecode:
                        {
                            if (receiveBytes.Length > 35)
                            {
                                var timecode = Encoding.UTF8.GetString(receiveBytes[28..39]);

                                if (DateTime.TryParse(timecode, out var timestamp))
                                {
                                    var totalSeconds = timestamp.Hour * 3600 + timestamp.Minute * 60 + timestamp.Second + (float)timestamp.Millisecond / 1000;
                                    m_Timecode.tc = Timecode.FromSeconds(FrameRate, totalSeconds);
                                    m_Timecode.rate = FrameRate;
                                }
                            }
                            break;
                        }
                    }
                }
            }
            catch (SocketException socketException)
            {
                Debug.LogError($"{nameof(XsensClient)}({port}): " + socketException.Message);
            }
            catch (System.IO.IOException ioException)
            {
                if (IsConnected)
                    Debug.LogError($"{nameof(XsensClient)}({port}): " + ioException.Message);
            }
            finally
            {
                m_Client?.Close();
                m_Client = null;
                IsConnected = false;
            }
        }

        FrameData ParsePacket(byte[] data)
        {
            XsDataPacket dataPacket = new XsQuaternionPacket(data);
            XsMvnPose pose = dataPacket.getPose();

            var frame = new FrameData();

            // Data Packet Type 2: Position and Quaternion Data for Each Body Segment.
            // Since there is no linear velocity or center of mass (COM) data, the corresponding arrays (Velocities and PositionsCOM) are initialized as empty. This allows us to differentiate between frames that contain only quaternion, velocity, or COM data.
            if (packetType == 2)
            {
                frame = new FrameData()
                {
                    TC = m_Timecode.tc,                 // Timecode
                    FrameRate = m_Timecode.rate,        // Frame rate
                    SegmentCount = data.Length / 32,    // Number of body segments (each segment's data is 32 bytes)
                    Positions = pose.positions,         // Position data for each segment
                    Orientations = pose.orientations,   // Quaternion data for each segment
                    Velocities = new Vector3[0],        // Velocity data for each segment (empty)
                    PositionsCOM = new Vector3[0],      // COM data (empty)
                    NumProps = pose.MvnCurrentPropCount // Number of additional properties
                };
                return frame;
            }
            // Data Packet Type 21: Linear Velocity Data for Each Body Segment.
            // Since only velocity data is received, the other arrays (Positions, Orientations, PositionsCOM) are initialized as empty.
            if (packetType == 21)
            {
                frame = new FrameData()
                {
                    TC = m_Timecode.tc, 
                    FrameRate = m_Timecode.rate, 
                    SegmentCount = data.Length / 32,               
                    Positions = new Vector3[0],
                    Orientations = new Quaternion[0],
                    Velocities = dataPacket.getConvertedData(),
                    PositionsCOM = new Vector3[0],
                    NumProps = 0
                };
                return frame;
            }
            // Data Packet Type 24: Center of Mass Position.
            // Since only COM data is received, the other arrays (Positions, Orientations, Velocities) are initialized as empty.
            if (packetType == 24)
            {
                frame = new FrameData()
                {
                    TC = m_Timecode.tc,        
                    FrameRate = m_Timecode.rate,          
                    SegmentCount = data.Length / 32,
                    Positions = new Vector3[0],
                    Orientations = new Quaternion[0],
                    Velocities = new Vector3[0],
                    PositionsCOM = dataPacket.getConvertedData(),
                    NumProps = 0
                };
                return frame;
            }

            return default;
        }
    }
}